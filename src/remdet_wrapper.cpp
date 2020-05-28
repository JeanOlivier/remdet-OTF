#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "remdet.hpp"


namespace py = pybind11;

template<typename T>
void declare_functions(py::module &m) {
    m.def("remdet", [](py::array_t<T, py::array::c_style>& array, uint64_t period) {
        auto buff = array.request();
        T *detpart = new T [period]();
        py::capsule free_when_done(detpart, [](void *f) {
            T *detpart = reinterpret_cast<T *>(f);
            delete[] detpart;
            });
        remdet((T*)buff.ptr, buff.size, detpart, period);
        return py::array_t<T>(
            {period,},      // shape
            {sizeof(T),},   // C-style contiguous strides for double
            detpart,        // the data pointer
            free_when_done);// numpy array references this parent
        }
    );
    m.def("getdet", [](py::array_t<T, py::array::c_style>& array, uint64_t period) {
        auto buff = array.request();

        T *detpart = new T [period]();
        py::capsule free_when_done(detpart, [](void *f) {
            T *detpart = reinterpret_cast<T *>(f);
            delete[] detpart;
            });
        getdet((T*)buff.ptr, buff.size, detpart, period);
        return py::array_t<T>(
            {period,},      // shape
            {sizeof(T),},   // C-style contiguous strides for double
            detpart,        // the data pointer
            free_when_done);// numpy array references this parent
        }
    );  
    m.def("deldet", [](py::array_t<T, py::array::c_style>& array, 
                       py::array_t<T, py::array::c_style>& detpart) {
        auto buff = array.request();
        auto dpart = detpart.request();
        deldet((T*)buff.ptr, buff.size, (T*)dpart.ptr, dpart.size);
        }
    );

}

PYBIND11_MODULE(remdet_wrapper, m) {
    m.doc() = "pybind11 wrapper for remdet"; // optional module docstring
    m.attr("the_answer") = 42;
    m.def("set_mpreal_precision", &set_mpreal_precision);
    m.def("set_num_threads", &omp_set_num_threads);
    
    set_mpreal_precision(100);  // Precision for all that's to follow

    declare_functions<int8_t>(m);
    declare_functions<uint8_t>(m);
    declare_functions<int16_t>(m);
    declare_functions<uint16_t>(m);
}

