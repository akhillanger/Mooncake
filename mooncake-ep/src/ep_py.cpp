#include <mooncake_ep_buffer.h>
#include <pybind11/gil.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace mooncake {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_ep_buffer_size_hint", &get_ep_buffer_size_hint);

    py::class_<EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait)
        .def("synchronize", &EventHandle::synchronize);

    m.attr("MAX_QP_COUNT") = pybind11::int_(MAX_QP_COUNT);

    py::class_<MooncakeEpBuffer>(m, "Buffer")
        .def(py::init<int, int, int64_t, bool>(), py::arg("rank"),
             py::arg("num_ranks"), py::arg("num_ep_buffer_bytes"),
             py::arg("use_nccl") = false)
        .def("ibgda_disabled", &MooncakeEpBuffer::ibgda_disabled)
        .def("nccl_enabled", &MooncakeEpBuffer::nccl_enabled)
        .def("get_nccl_unique_id_size",
             &MooncakeEpBuffer::get_nccl_unique_id_size)
        .def("get_nccl_unique_id", &MooncakeEpBuffer::get_nccl_unique_id)
        .def("initialize_nccl", &MooncakeEpBuffer::initialize_nccl)
        .def("get_nccl_properties", &MooncakeEpBuffer::get_nccl_properties)
        .def("use_fast_path", &MooncakeEpBuffer::use_fast_path)
        .def("update_local_qpns", &MooncakeEpBuffer::update_local_qpns)
        .def("is_roce", &MooncakeEpBuffer::is_roce)
        .def("sync_ibgda_peers", &MooncakeEpBuffer::sync_ibgda_peers)
        .def("get_mr_info", &MooncakeEpBuffer::get_mr_info)
        .def("get_gid", &MooncakeEpBuffer::get_gid)
        .def("get_local_qpns", &MooncakeEpBuffer::get_local_qpns)
        .def("get_local_lids", &MooncakeEpBuffer::get_local_lids)
        .def("get_ipc_handle", &MooncakeEpBuffer::get_ipc_handle)
        .def("sync_nvlink_ipc_handles",
             &MooncakeEpBuffer::sync_nvlink_ipc_handles)
        .def("dispatch", &MooncakeEpBuffer::dispatch)
        .def("combine", &MooncakeEpBuffer::combine)
        .def("get_next_combine_buffer",
             &MooncakeEpBuffer::get_next_combine_buffer);
}

}  // namespace mooncake
