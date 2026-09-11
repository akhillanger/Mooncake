// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <thread>

#include "control_plane/coordinator.h"
#include "comm_types.h"
#include "error_types.h"

namespace mooncake {
namespace {

class NcclFailureControlTest : public ::testing::Test {
   protected:
    CentralizedCoordinatorStateMachine coordinator{
        4, std::chrono::milliseconds(50), std::chrono::milliseconds(100)};
    // Root is not global rank zero.
    const std::vector<GlobalRank> members{1, 2, 3};

    static uint64_t session(int rank) { return 100 + rank; }

    RegisterAgentResponse snapshot(int rank = 1) {
        return coordinator
            .handleRegisterAgent(
                {.rank = rank, .agent_session_id = session(rank)})
            .response;
    }

    void SetUp() override {
        for (int rank = 0; rank < 4; ++rank) {
            ASSERT_TRUE(snapshot(rank).success);
        }
        const auto epochs = snapshot().all_rank_epochs;
        for (int rank = 0; rank < 4; ++rank) {
            coordinator.handleLinkEventReport(
                {.reporter_rank = rank,
                 .agent_session_id = session(rank),
                 .reporter_rank_epoch = epochs[rank],
                 .report_id = 1,
                 .events = std::vector<LinkEvent::EventType>(
                     4, LinkEvent::EventType::Success),
                 .target_rank_epochs = epochs});
        }
    }

    GroupView view(const GroupId& id) {
        for (const auto& group : snapshot().groups) {
            if (group.group_id == id) return group;
        }
        throw std::runtime_error("test group not found");
    }

    NcclCollectiveFailure createGroup(const std::string& bootstrap,
                                      uint8_t token, bool nccl = true) {
        GroupId id;
        for (int rank : members) {
            const auto registration = coordinator.handleRegisterGroup(
                {.rank = rank,
                 .agent_session_id = session(rank),
                 .group_bootstrap_id = bootstrap,
                 .max_group_size = 3,
                 .rank_order = members});
            PG_ASSERT(registration.response.success,
                      "test registration failed");
            id = registration.response.view.group_id;
        }
        NcclCollectiveFailure failure{.group_id = id};
        failure.unique_id.fill(token);
        for (int rank : members) {
            GroupEndpointInfo endpoint;
            endpoint.nccl_collectives_enabled = nccl;
            if (rank == members.front() && nccl) {
                endpoint.nccl_unique_id = failure.unique_id;
                endpoint.nccl_unique_id_size = kNcclUniqueIdBytes;
            }
            PG_ASSERT(
                coordinator
                    .handlePublishEndpoint({.rank = rank,
                                            .agent_session_id = session(rank),
                                            .endpoints = {{id, endpoint}}})
                    .response.success,
                "test endpoint publication failed");
        }
        const auto bootstrapping = view(id);
        PG_ASSERT(bootstrapping.status == GroupStatus::BootstrapSyncing,
                  "test group did not reach bootstrap barrier");
        for (int rank : members) {
            coordinator.handleViewUpdateAck(id, rank, bootstrapping.epoch,
                                            true);
        }
        PG_ASSERT(view(id).status == GroupStatus::Ready,
                  "test group not ready");
        return failure;
    }

    auto heartbeat(int rank, std::vector<NcclCollectiveFailure> failures = {}) {
        return coordinator.handleHeartbeat(
            {.rank = rank,
             .agent_session_id = session(rank),
             .nccl_failures = std::move(failures)});
    }

    auto recover(uint64_t id, int rank, const NcclCollectiveFailure& failure,
                 int te_tasks = 3) {
        const auto epoch = view(failure.group_id).epoch;
        return coordinator.handleSyncAfterFailure(
            id,
            {.group_id = failure.group_id,
             .reporter_rank = rank,
             .agent_session_id = session(rank),
             .current_epoch = epoch,
             .nccl_recovery = NcclRecoveryRequest{failure, epoch, te_tasks}});
    }
};

TEST_F(NcclFailureControlTest,
       RecoveryWaitsForEveryActiveRankAndRetriesCommit) {
    auto failure = createGroup("device:recovery", 7);
    failure.first_failed_operation = 4;
    const auto before = snapshot();
    EXPECT_TRUE(recover(11, 1, failure).effects.empty());
    EXPECT_TRUE(recover(12, 2, failure).effects.empty());
    // A duplicate acknowledgement cannot replace the idle rank.
    EXPECT_TRUE(recover(13, 1, failure).effects.empty());
    failure.first_failed_operation = 3;
    auto commit = recover(14, 3, failure);
    ASSERT_EQ(commit.effects.size(), 4);
    for (const auto& effect : commit.effects) {
        const auto& response = std::get<ReplySync>(effect).response;
        EXPECT_EQ(response.status, SyncAfterFailureStatus::Reconciled);
        ASSERT_TRUE(response.nccl_recovery);
        EXPECT_EQ(*response.nccl_recovery, failure);
    }
    // Model a lost commit reply and retry after the admission deadline.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    coordinator.tick();
    const auto retry = recover(15, 1, failure);
    ASSERT_EQ(retry.effects.size(), 1);
    EXPECT_EQ(std::get<ReplySync>(retry.effects.front()).response.nccl_recovery,
              failure);
    const auto malformed_retry = recover(16, 1, failure, 4);
    ASSERT_EQ(malformed_retry.effects.size(), 1);
    EXPECT_EQ(
        std::get<ReplySync>(malformed_retry.effects.front()).response.status,
        SyncAfterFailureStatus::Rejected);
    const auto valid_retry = recover(17, 1, failure);
    ASSERT_EQ(valid_retry.effects.size(), 1);
    EXPECT_EQ(
        std::get<ReplySync>(valid_retry.effects.front()).response.nccl_recovery,
        failure);
    EXPECT_EQ(snapshot().groups, before.groups);
    EXPECT_EQ(snapshot().all_rank_states, before.all_rank_states);
}

TEST_F(NcclFailureControlTest, RecoveryTimeoutRejectsWithoutRemovingRanks) {
    const auto failure = createGroup("device:recovery", 7);
    const auto before = snapshot();
    EXPECT_TRUE(recover(11, 1, failure).effects.empty());
    EXPECT_TRUE(recover(12, 2, failure).effects.empty());
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const auto expired = coordinator.tick();
    ASSERT_EQ(expired.effects.size(), 2);
    for (const auto& effect : expired.effects) {
        const auto& response = std::get<ReplySync>(effect).response;
        EXPECT_EQ(response.status, SyncAfterFailureStatus::Rejected);
        EXPECT_FALSE(response.nccl_recovery);
        EXPECT_NE(response.reject_reason.find("timed out"), std::string::npos);
    }
    EXPECT_EQ(snapshot().groups, before.groups);
    EXPECT_EQ(snapshot().all_rank_states, before.all_rank_states);
    // Expiry drops old acknowledgements: all ranks must arrive again.
    EXPECT_TRUE(recover(13, 3, failure).effects.empty());
    EXPECT_TRUE(recover(14, 1, failure).effects.empty());
    EXPECT_EQ(recover(15, 2, failure).effects.size(), 3);
}

TEST_F(NcclFailureControlTest, RecoveryRejectsDifferentTESequences) {
    const auto failure = createGroup("device:recovery", 7);
    EXPECT_TRUE(recover(11, 1, failure, 3).effects.empty());
    const auto mismatch = recover(12, 2, failure, 4);
    ASSERT_EQ(mismatch.effects.size(), 2);
    for (const auto& effect : mismatch.effects) {
        const auto& response = std::get<ReplySync>(effect).response;
        EXPECT_EQ(response.status, SyncAfterFailureStatus::Rejected);
        EXPECT_FALSE(response.nccl_recovery);
    }
}

TEST_F(NcclFailureControlTest, RecoveryRejectsStaleAndNonmemberRequests) {
    const auto failure = createGroup("device:recovery", 7);
    auto wrong = failure;
    wrong.unique_id[0]++;
    for (const auto& result :
         {recover(11, 0, failure), recover(12, 2, wrong)}) {
        ASSERT_EQ(result.effects.size(), 1);
        EXPECT_EQ(std::get<ReplySync>(result.effects.front()).response.status,
                  SyncAfterFailureStatus::Rejected);
    }
    const auto epoch = view(failure.group_id).epoch;
    for (bool stale_session : {true, false}) {
        const auto result = coordinator.handleSyncAfterFailure(
            13, {.group_id = failure.group_id,
                 .reporter_rank = 1,
                 .agent_session_id = session(1) - (stale_session ? 1 : 0),
                 .current_epoch = epoch - (stale_session ? 0 : 1),
                 .nccl_recovery = NcclRecoveryRequest{failure, epoch, 3}});
        ASSERT_EQ(result.effects.size(), 1);
        EXPECT_EQ(std::get<ReplySync>(result.effects.front()).response.status,
                  SyncAfterFailureStatus::Rejected);
    }
}

TEST_F(NcclFailureControlTest, RecoveryDoesNotCommitAfterMembershipChange) {
    const auto failure = createGroup("device:recovery", 7);
    EXPECT_TRUE(recover(11, 1, failure).effects.empty());
    EXPECT_TRUE(recover(12, 2, failure).effects.empty());
    coordinator.handleUnregisterGroup({failure.group_id, 3, session(3)});
    const auto result = coordinator.tick();
    size_t rejected = 0;
    for (const auto& effect : result.effects) {
        if (const auto* reply = std::get_if<ReplySync>(&effect)) {
            EXPECT_EQ(reply->response.status, SyncAfterFailureStatus::Rejected);
            EXPECT_FALSE(reply->response.nccl_recovery);
            ++rejected;
        }
    }
    EXPECT_EQ(rejected, 2);
}

TEST_F(NcclFailureControlTest, RecoveryRejectsDepartedSession) {
    const auto failure = createGroup("device:recovery", 7);
    EXPECT_TRUE(recover(11, 1, failure).effects.empty());
    EXPECT_TRUE(recover(12, 2, failure).effects.empty());
    coordinator.handleUnregisterAgent(
        {.rank = 2, .agent_session_id = session(2)});
    const auto result = coordinator.tick();
    size_t rejected = 0;
    for (const auto& effect : result.effects) {
        if (const auto* reply = std::get_if<ReplySync>(&effect)) {
            EXPECT_EQ(reply->response.status, SyncAfterFailureStatus::Rejected);
            ++rejected;
        }
    }
    EXPECT_EQ(rejected, 2);
}

TEST_F(NcclFailureControlTest, RecoveryRepliesBeforeCoordinatorShutdown) {
    const auto failure = createGroup("device:recovery", 7);
    EXPECT_TRUE(recover(11, 1, failure).effects.empty());
    coordinator.requestShutdown();
    size_t rejected = 0;
    bool shutdown = false;
    for (int rank = 0; rank < 4; ++rank) {
        const auto result = coordinator.handleUnregisterAgent(
            {.rank = rank, .agent_session_id = session(rank)});
        for (const auto& effect : result.effects) {
            EXPECT_FALSE(shutdown);
            if (const auto* reply = std::get_if<ReplySync>(&effect)) {
                EXPECT_EQ(reply->response.status,
                          SyncAfterFailureStatus::Rejected);
                ++rejected;
            }
            shutdown = std::holds_alternative<ShutdownCoordinatorHost>(effect);
        }
    }
    // Shutdown is confirmed by the periodic tick, not unregisterAgent itself.
    for (const auto& effect : coordinator.tick().effects) {
        EXPECT_FALSE(shutdown);
        if (const auto* reply = std::get_if<ReplySync>(&effect)) {
            EXPECT_EQ(reply->response.status, SyncAfterFailureStatus::Rejected);
            ++rejected;
        }
        shutdown = std::holds_alternative<ShutdownCoordinatorHost>(effect);
    }
    EXPECT_TRUE(shutdown);
    EXPECT_EQ(rejected, 1);
}

TEST_F(NcclFailureControlTest, RepeatsNotificationsWithoutChangingMembership) {
    const auto failure = createGroup("device:failed", 7);
    const auto before = snapshot();
    auto first = heartbeat(2, {failure});
    EXPECT_TRUE(first.effects.empty());
    EXPECT_EQ(first.response.nccl_failures,
              std::vector<NcclCollectiveFailure>{failure});
    // Discard the first response, then retry the same report and poll without
    // reporting. Both request loss and response loss are covered by repetition.
    EXPECT_EQ(heartbeat(2, {failure, failure}).response.nccl_failures,
              first.response.nccl_failures);
    for (int rank : members) {
        EXPECT_EQ(heartbeat(rank).response.nccl_failures,
                  first.response.nccl_failures);
    }
    EXPECT_TRUE(heartbeat(0).response.nccl_failures.empty());
    const auto after = snapshot();
    EXPECT_EQ(after.groups, before.groups);
    EXPECT_EQ(after.all_rank_states, before.all_rank_states);
    EXPECT_EQ(after.all_rank_state_versions, before.all_rank_state_versions);
    const auto sync = coordinator.handleSyncAfterFailure(
        1, {.group_id = failure.group_id,
            .reporter_rank = 2,
            .agent_session_id = session(2),
            .current_epoch = view(failure.group_id).epoch});
    ASSERT_EQ(sync.effects.size(), 1);
    EXPECT_EQ(std::get<ReplySync>(sync.effects.front()).response.status,
              SyncAfterFailureStatus::NoPending);
}

TEST_F(NcclFailureControlTest, RejectsStaleSessionTokenAndNonmember) {
    const auto failure = createGroup("device:failed", 7);
    const auto stale =
        coordinator.handleHeartbeat({.rank = 2,
                                     .agent_session_id = session(2) - 1,
                                     .nccl_failures = {failure}});
    EXPECT_TRUE(stale.response.require_new_session);
    EXPECT_TRUE(stale.response.nccl_failures.empty());
    auto wrong_token = failure;
    wrong_token.unique_id[0]++;
    auto wrong_group = failure;
    wrong_group.group_id = "missing";
    EXPECT_TRUE(heartbeat(2, {wrong_token, wrong_group})
                    .response.nccl_failures.empty());
    EXPECT_TRUE(heartbeat(0, {failure}).response.nccl_failures.empty());
    EXPECT_TRUE(heartbeat(1).response.nccl_failures.empty());
    coordinator.handleUnregisterGroup({failure.group_id, 2, session(2)});
    EXPECT_TRUE(heartbeat(2, {failure}).response.nccl_failures.empty());
    EXPECT_TRUE(heartbeat(1).response.nccl_failures.empty());
}

TEST_F(NcclFailureControlTest, IgnoresTransferEngineGroups) {
    const auto failure = createGroup("device:te", 7, false);
    EXPECT_TRUE(heartbeat(2, {failure}).response.nccl_failures.empty());
    EXPECT_TRUE(heartbeat(1).response.nccl_failures.empty());
}

TEST_F(NcclFailureControlTest, OldTokenCannotOverwriteNewGenerationFailure) {
    const auto old = createGroup("device:failed", 7);
    auto next = old;
    next.unique_id.fill(8);
    auto endpoint = *view(old.group_id).members[1].endpoint;
    endpoint.nccl_unique_id = next.unique_id;
    ASSERT_TRUE(
        coordinator
            .handlePublishEndpoint({.rank = 1,
                                    .agent_session_id = session(1),
                                    .endpoints = {{old.group_id, endpoint}}})
            .response.success);
    EXPECT_EQ(heartbeat(2, {next}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{next});
    EXPECT_EQ(heartbeat(3, {old}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{next});
}

TEST_F(NcclFailureControlTest, KeepsEarliestFailedOperationAcrossReports) {
    auto later = createGroup("device:failed", 7);
    later.first_failed_operation = 12;
    auto earlier = later;
    earlier.first_failed_operation = 10;
    EXPECT_EQ(heartbeat(2, {later}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{later});
    EXPECT_EQ(heartbeat(3, {earlier}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{earlier});
    EXPECT_EQ(heartbeat(2, {later}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{earlier});
}

TEST(NcclFailureStatusTest,
     LateNotificationUpdatesRetainedWorkOnlyFromFailure) {
    auto state = std::make_shared<GpuCollectiveFailureState>();
    GpuCollectiveStatus earlier, affected, subsequent;
    earlier.failure_state = affected.failure_state = subsequent.failure_state =
        state;
    earlier.sequence = 9;
    affected.sequence = 10;
    subsequent.sequence = 11;
    EXPECT_FALSE(affected.isAborted());
    state->failFrom(10);
    state->failFrom(
        12);        // An out-of-order report must not erase earlier failure.
    state.reset();  // Status remains valid after executor ownership is gone.
    EXPECT_FALSE(earlier.isAborted());
    EXPECT_TRUE(affected.isAborted());
    EXPECT_TRUE(subsequent.isAborted());
}

TEST_F(NcclFailureControlTest, GroupDestructionClearsLatchedFailure) {
    const auto failure = createGroup("device:reused", 7);
    const auto other = createGroup("device:other", 9);
    EXPECT_EQ(heartbeat(2, {failure}).response.nccl_failures,
              std::vector<NcclCollectiveFailure>{failure});
    EXPECT_NE(other.group_id, failure.group_id);
    for (int rank : members) {
        ASSERT_TRUE(
            coordinator
                .handleUnregisterGroup({failure.group_id, rank, session(rank)})
                .response.success);
    }
    // Even reusing the bootstrap name and token cannot target the new runtime
    // id.
    const auto replacement = createGroup("device:reused", 7);
    EXPECT_NE(replacement.group_id, failure.group_id);
    EXPECT_TRUE(heartbeat(2, {failure}).response.nccl_failures.empty());
    EXPECT_TRUE(heartbeat(1).response.nccl_failures.empty());
}

}  // namespace
}  // namespace mooncake
