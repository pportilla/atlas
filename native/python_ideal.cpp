// Optional CPython stable-ABI owner of the same kernel handles used by ctypes.
#define Py_LIMITED_API 0x030a0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <cstring>
#include "prepared_ideal.hpp"
extern "C" void atlas_groebner_release(char*) noexcept;

namespace {
struct Handle {
    PyObject_HEAD
    void* ideal;
    PyObject* values;  // Optional tuple of immutable session expressions.
};

PyObject* owned(PyTypeObject* type, void* ideal) {
    if (!ideal) {
        const char* error = atlas_ideal_error();
        if (*error) { PyErr_SetString(PyExc_ArithmeticError,error); return nullptr; }
        Py_RETURN_NONE;  // Retry in the cancellable worker.
    }
    auto* self = reinterpret_cast<Handle*>(PyType_GenericAlloc(type,0));
    if (!self) { atlas_ideal_release(ideal); return nullptr; }
    self->ideal = ideal;
    return reinterpret_cast<PyObject*>(self);
}

int traverse(Handle* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(self->values);
    return 0;
}
int clear(Handle* self) { Py_CLEAR(self->values); return 0; }
void destroy(Handle* self) {
    PyObject_GC_UnTrack(self);
    clear(self);
    atlas_ideal_release(self->ideal);
    PyTypeObject* type = Py_TYPE(self);
    PyObject_GC_Del(self);
    Py_DECREF(type);
}
PyObject* reject_new(PyTypeObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_TypeError,"Prepare an ideal from its generators.");
    return nullptr;
}

template<bool product>
PyObject* binary(Handle* self, PyObject* right) {
    if (Py_TYPE(self) != Py_TYPE(right)) {
        PyErr_SetString(PyExc_TypeError,"Expected another prepared ideal.");
        return nullptr;
    }
    void* answer;
    Py_BEGIN_ALLOW_THREADS
    answer = atlas_ideal_binary(self->ideal,reinterpret_cast<Handle*>(right)->ideal,product);
    Py_END_ALLOW_THREADS
    return owned(Py_TYPE(self),answer);
}
PyObject* power(Handle* self, PyObject* args) {
    PyObject* exponent;
    int lex;
    if (!PyArg_ParseTuple(args,"Op",&exponent,&lex)) return nullptr;
    unsigned long degree = PyLong_AsUnsignedLong(exponent);
    if (PyErr_Occurred()) return nullptr;
    void* answer;
    Py_BEGIN_ALLOW_THREADS
    answer = atlas_ideal_power_small(self->ideal,degree,lex);
    Py_END_ALLOW_THREADS
    return owned(Py_TYPE(self),answer);
}
PyObject* size(Handle* self, PyObject*) {
    size_t count;
    Py_BEGIN_ALLOW_THREADS
    count = atlas_ideal_size(self->ideal);
    Py_END_ALLOW_THREADS
    if (*atlas_ideal_error()) { PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error()); return nullptr; }
    return PyLong_FromSize_t(count);
}
PyObject* lines(char* text) {
    if (!text) { PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error()); return nullptr; }
    auto* value = PyUnicode_FromString(text);
    atlas_groebner_release(text);
    if (!value) return nullptr;
    auto* result = PyUnicode_Splitlines(value,0);
    Py_DECREF(value);
    return result;
}
PyObject* rows(Handle* self, PyObject*) {
    char* text;
    Py_BEGIN_ALLOW_THREADS
    text = atlas_ideal_rows(self->ideal);
    Py_END_ALLOW_THREADS
    return lines(text);
}
PyObject* expressions(Handle* self, PyObject* names) {
    auto* separator = PyUnicode_FromString(" ");
    if (!separator) return nullptr;
    auto* joined = PyUnicode_Join(separator,names);
    Py_DECREF(separator);
    if (!joined) return nullptr;
    const char* names_text = PyUnicode_AsUTF8AndSize(joined,nullptr);
    if (!names_text) { Py_DECREF(joined); return nullptr; }
    char* text;
    Py_BEGIN_ALLOW_THREADS
    text = atlas_ideal_expressions(self->ideal,names_text);
    Py_END_ALLOW_THREADS
    Py_DECREF(joined);
    return lines(text);
}

PyObject* prepare(PyObject* type, PyObject* args);
PyObject* copy_handle(PyObject* self, PyObject*) { return Py_NewRef(self); }
PyMethodDef methods[] = {
    {"prepare", prepare, METH_CLASS | METH_VARARGS, nullptr},
    {"__copy__", copy_handle, METH_NOARGS, nullptr},
    {"__deepcopy__", copy_handle, METH_O, nullptr},
    {"sum", reinterpret_cast<PyCFunction>(binary<false>), METH_O, nullptr},
    {"product", reinterpret_cast<PyCFunction>(binary<true>), METH_O, nullptr},
    {"power", reinterpret_cast<PyCFunction>(power), METH_VARARGS, nullptr},
    {"size", reinterpret_cast<PyCFunction>(size), METH_NOARGS, nullptr},
    {"rows", reinterpret_cast<PyCFunction>(rows), METH_NOARGS, nullptr},
    {"expressions", reinterpret_cast<PyCFunction>(expressions), METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
PyMemberDef members[] = {
    {"_values", T_OBJECT_EX, offsetof(Handle,values), 0, nullptr},
    {nullptr,0,0,0,nullptr}
};
PyType_Slot slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(reject_new)},
    {Py_tp_dealloc, reinterpret_cast<void*>(destroy)},
    {Py_tp_traverse, reinterpret_cast<void*>(traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(clear)},
    {Py_tp_members, members}, {Py_tp_methods, methods}, {0,nullptr}
};
PyType_Spec spec = {"_atlas_ideal.Ideal",sizeof(Handle),0,Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,slots};

PyObject* prepare(PyObject* type, PyObject* args) {
    Py_ssize_t width, count, bytes;
    PyObject* payload;
    if (!PyArg_ParseTuple(args,"nO!n",&width,&PyBytes_Type,&payload,&count)) return nullptr;
    char* rows;
    if (PyBytes_AsStringAndSize(payload,&rows,&bytes) < 0) return nullptr;
    if (width < 0 || count < 0 || std::memchr(rows,'\0',static_cast<size_t>(bytes))) {
        PyErr_SetString(PyExc_ValueError,"Invalid prepared ideal dimensions or rows.");
        return nullptr;
    }
    void* ideal;
    Py_BEGIN_ALLOW_THREADS
    ideal = atlas_ideal_prepare(static_cast<size_t>(width),rows,static_cast<size_t>(count));
    Py_END_ALLOW_THREADS
    return owned(reinterpret_cast<PyTypeObject*>(type),ideal);
}
PyModuleDef module = {PyModuleDef_HEAD_INIT,"_atlas_ideal",nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__atlas_ideal() {
    auto* m = PyModule_Create(&module);
    if (!m) return nullptr;
    auto* type = PyType_FromSpec(&spec);
    if (!type || PyModule_AddObject(m,"Ideal",type) < 0) { Py_XDECREF(type); Py_DECREF(m); return nullptr; }
    return m;
}
