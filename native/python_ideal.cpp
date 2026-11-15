// Optional CPython stable-ABI owner of the same kernel handles used by ctypes.
#define Py_LIMITED_API 0x030a0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <cstring>
#include "prepared_ideal.hpp"
extern "C" void atlas_groebner_release(char*) noexcept;

namespace {
constexpr char ring_name[] = "atlas.ideal_ring";
constexpr char polynomial_name[] = "atlas.ideal_polynomial";

void destroy_ring(PyObject* obj) { atlas_ideal_ring_release(PyCapsule_GetPointer(obj,ring_name)); }
void destroy_polynomial(PyObject* obj) { atlas_ideal_polynomial_release(PyCapsule_GetPointer(obj,polynomial_name)); }

PyObject* make_ring(PyObject*, PyObject* width) {
    const size_t n = PyLong_AsSize_t(width);
    if (PyErr_Occurred()) return nullptr;
    auto* ring = atlas_ideal_ring(n);
    if (!ring) {
        PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error());
        return nullptr;
    }
    auto* result = PyCapsule_New(ring,ring_name,destroy_ring);
    if (!result) atlas_ideal_ring_release(ring);
    return result;
}

PyObject* make_polynomial(PyObject*, PyObject* args) {
    PyObject *parent, *data;
    char* row;
    Py_ssize_t length;
    if (!PyArg_ParseTuple(args,"OS",&parent,&data) || PyBytes_AsStringAndSize(data,&row,&length) < 0)
        return nullptr;
    auto* ring = PyCapsule_GetPointer(parent,ring_name);
    if (!ring) return nullptr;
    if (std::memchr(row,'\0',static_cast<size_t>(length))) {
        PyErr_SetString(PyExc_ValueError,"Invalid polynomial row.");
        return nullptr;
    }
    void* polynomial;
    Py_BEGIN_ALLOW_THREADS
    polynomial = atlas_ideal_polynomial(ring,row);
    Py_END_ALLOW_THREADS
    if (!polynomial) {
        PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error());
        return nullptr;
    }
    auto* result = PyCapsule_New(polynomial,polynomial_name,destroy_polynomial);
    if (!result) atlas_ideal_polynomial_release(polynomial);
    return result;
}

PyObject* polynomial_row(PyObject*, PyObject* obj) {
    auto* poly = PyCapsule_GetPointer(obj,polynomial_name);
    if (!poly) return nullptr;
    char* row;
    Py_BEGIN_ALLOW_THREADS
    row = atlas_ideal_polynomial_row(poly);
    Py_END_ALLOW_THREADS
    if (!row) { PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error()); return nullptr; }
    auto* bytes = PyBytes_FromString(row);
    atlas_groebner_release(row);
    return bytes;
}

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

PyObject* from_polynomials(PyObject* type, PyObject* args) {
    PyObject *parent, *items;
    if (!PyArg_ParseTuple(args,"OO!",&parent,&PyTuple_Type,&items)) return nullptr;
    auto* ring = PyCapsule_GetPointer(parent,ring_name);
    if (!ring) return nullptr;
    const auto count = PyTuple_Size(items);
    if (count > 128) Py_RETURN_NONE;
    void* polynomials[128];
    for (Py_ssize_t i = 0; i < count; ++i) {
        polynomials[i] = PyCapsule_GetPointer(PyTuple_GetItem(items,i),polynomial_name);
        if (!polynomials[i]) return nullptr;
    }
    // This bounded call only retains references to immutable polynomials.
    auto* result = atlas_ideal_from_polynomials(ring,polynomials,static_cast<size_t>(count));
    return owned(reinterpret_cast<PyTypeObject*>(type),result);
}

struct Fields {
    PyObject *parent_name, *polynomial_name;
};
struct PolynomialOwners {
    PyObject* objects[128];
    size_t count = 0;
    ~PolynomialOwners() { while (count) Py_DECREF(objects[--count]); }
};

PyObject* declare(PyObject* type, PyObject* args) {
    PyObject *parent_token, *parent, *polynomial_type, *items;
    if (!PyArg_ParseTuple(args,"OOOO!",&parent_token,&parent,&polynomial_type,&PyTuple_Type,&items))
        return nullptr;
    auto* ring = PyCapsule_GetPointer(parent_token,ring_name);
    if (!ring) return nullptr;
    const auto count = PyTuple_Size(items);
    if (count > 128) Py_RETURN_NOTIMPLEMENTED; // The general path validates every parent.
    const auto* fields = static_cast<const Fields*>(PyType_GetModuleState(reinterpret_cast<PyTypeObject*>(type)));
    if (!fields) return nullptr;
    void* polynomials[128];
    PolynomialOwners owners;
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* f = PyTuple_GetItem(items,i);
        if (reinterpret_cast<PyObject*>(Py_TYPE(f)) != polynomial_type) Py_RETURN_NOTIMPLEMENTED;
        auto* space = PyObject_GetAttr(f,fields->parent_name);
        if (!space) return nullptr;
        bool same = space == parent;
        Py_DECREF(space);
        if (!same) Py_RETURN_NOTIMPLEMENTED; // General path can embed a QQ scalar.
        auto* owner = PyObject_GetAttr(f,fields->polynomial_name);
        if (!owner) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError)) return nullptr;
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        owners.objects[owners.count++] = owner;
        if (!PyCapsule_IsValid(owner,polynomial_name)) Py_RETURN_NOTIMPLEMENTED;
        polynomials[i] = PyCapsule_GetPointer(owner,polynomial_name);
    }
    return owned(reinterpret_cast<PyTypeObject*>(type),
                 atlas_ideal_from_polynomials(ring,polynomials,static_cast<size_t>(count)));
}

PyObject* copy_handle(PyObject* self, PyObject*) { return Py_NewRef(self); }
PyMethodDef methods[] = {
    {"prepare", prepare, METH_CLASS | METH_VARARGS, nullptr},
    {"ring", make_ring, METH_CLASS | METH_O, nullptr},
    {"polynomial", make_polynomial, METH_CLASS | METH_VARARGS, nullptr},
    {"polynomial_row", polynomial_row, METH_CLASS | METH_O, nullptr},
    {"from_polynomials", from_polynomials, METH_CLASS | METH_VARARGS, nullptr},
    {"declare", declare, METH_CLASS | METH_VARARGS, nullptr},
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
int module_traverse(PyObject* m, visitproc visit, void* arg) {
    auto* fields = static_cast<Fields*>(PyModule_GetState(m));
    Py_VISIT(fields->parent_name);
    Py_VISIT(fields->polynomial_name);
    return 0;
}
int module_clear(PyObject* m) {
    auto* fields = static_cast<Fields*>(PyModule_GetState(m));
    Py_CLEAR(fields->parent_name);
    Py_CLEAR(fields->polynomial_name);
    return 0;
}
PyModuleDef module = {PyModuleDef_HEAD_INIT,"_atlas_ideal",nullptr,sizeof(Fields),nullptr,nullptr,
                      module_traverse,module_clear,nullptr};
}
PyMODINIT_FUNC PyInit__atlas_ideal() {
    auto* m = PyModule_Create(&module);
    if (!m) return nullptr;
    auto* fields = static_cast<Fields*>(PyModule_GetState(m));
    fields->parent_name = PyUnicode_InternFromString("_space");
    fields->polynomial_name = PyUnicode_InternFromString("_native_polynomial");
    if (!fields->parent_name || !fields->polynomial_name) { Py_DECREF(m); return nullptr; }
    auto* type = PyType_FromModuleAndSpec(m,&spec,nullptr);
    if (!type || PyModule_AddObject(m,"Ideal",type) < 0) { Py_XDECREF(type); Py_DECREF(m); return nullptr; }
    return m;
}
