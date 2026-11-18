// Optional CPython stable-ABI owner of the same kernel handles used by ctypes.
#define Py_LIMITED_API 0x030a0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <cstring>
#include "prepared_ideal.hpp"
extern "C" void atlas_groebner_release(char*) noexcept;
int atlas_add_session_runner(PyObject* module);

namespace {
constexpr char ring_name[] = "atlas.ideal_ring";
constexpr char polynomial_name[] = "atlas.ideal_polynomial";

void destroy_ring(PyObject* obj) { atlas_ideal_ring_release(PyCapsule_GetPointer(obj,ring_name)); }
void destroy_polynomial(PyObject* obj) {
    atlas_ideal_polynomial_release(PyCapsule_GetPointer(obj,polynomial_name));
    Py_XDECREF(static_cast<PyObject*>(PyCapsule_GetContext(obj)));
}

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

PyObject* owned_polynomial(void* polynomial, PyObject* parent) {
    auto* result = PyCapsule_New(polynomial,polynomial_name,destroy_polynomial);
    if (result) {
        // A generator carries its exact ring through the C++ and Python calls.
        if (PyCapsule_SetContext(result,parent) < 0) { Py_DECREF(result); return nullptr; }
        Py_INCREF(parent);
    } else atlas_ideal_polynomial_release(polynomial);
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
    return owned_polynomial(polynomial,parent);
}

PyObject* polynomial_gcd(PyObject*, PyObject* args) {
    PyObject *left, *right;
    int lex;
    if (!PyArg_ParseTuple(args,"OOp",&left,&right,&lex)) return nullptr;
    auto* f = PyCapsule_GetPointer(left,polynomial_name);
    if (!f) return nullptr;
    auto* g = PyCapsule_GetPointer(right,polynomial_name);
    if (!g) return nullptr;
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = atlas_ideal_polynomial_gcd(f,g,lex);
    Py_END_ALLOW_THREADS
    if (!result) {
        if (!*atlas_ideal_error()) Py_RETURN_NONE;
        PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error());
        return nullptr;
    }
    return owned_polynomial(result,static_cast<PyObject*>(PyCapsule_GetContext(left)));
}

PyObject* polynomial_is_zero(PyObject*, PyObject* obj) {
    auto* f = PyCapsule_GetPointer(obj,polynomial_name);
    if (!f) return nullptr;
    const int result = atlas_ideal_polynomial_is_zero(f);
    if (result < 0) { PyErr_SetString(PyExc_ArithmeticError,atlas_ideal_error()); return nullptr; }
    return PyBool_FromLong(result);
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
    PyTypeObject* handle_type; // Held alive by the instance's type or its bases.
    PyObject *space, *generators, *order; // Present for a public declared ideal.
};

PyObject* owned(PyTypeObject* type, void* ideal, PyTypeObject* instance_type = nullptr) {
    if (!ideal) {
        const char* error = atlas_ideal_error();
        if (*error) { PyErr_SetString(PyExc_ArithmeticError,error); return nullptr; }
        Py_RETURN_NONE;  // Retry in the cancellable worker.
    }
    auto* self = reinterpret_cast<Handle*>(PyType_GenericAlloc(instance_type ? instance_type : type,0));
    if (!self) { atlas_ideal_release(ideal); return nullptr; }
    self->ideal = ideal;
    self->handle_type = type;
    return reinterpret_cast<PyObject*>(self);
}

int traverse(Handle* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(self->values);
    Py_VISIT(self->space);
    Py_VISIT(self->generators);
    Py_VISIT(self->order);
    return 0;
}
int clear(Handle* self) {
    Py_CLEAR(self->values);
    Py_CLEAR(self->space);
    Py_CLEAR(self->generators);
    Py_CLEAR(self->order);
    return 0;
}
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
    if (!PyObject_TypeCheck(right,self->handle_type)) {
        PyErr_SetString(PyExc_TypeError,"Expected another prepared ideal.");
        return nullptr;
    }
    void* answer;
    Py_BEGIN_ALLOW_THREADS
    answer = atlas_ideal_binary(self->ideal,reinterpret_cast<Handle*>(right)->ideal,product);
    Py_END_ALLOW_THREADS
    return owned(self->handle_type,answer);
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
    return owned(self->handle_type,answer);
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

struct OwnedReference {
    PyObject* value;
    ~OwnedReference() { Py_XDECREF(value); }
};

struct ConstructorConfig { PyObject* definition; };
int traverse_constructor(PyObject* module, visitproc visit, void* arg) {
    Py_VISIT(static_cast<ConstructorConfig*>(PyModule_GetState(module))->definition);
    return 0;
}
int clear_constructor(PyObject* module) {
    Py_CLEAR(static_cast<ConstructorConfig*>(PyModule_GetState(module))->definition);
    return 0;
}
PyModuleDef constructor_module = {PyModuleDef_HEAD_INIT,"_atlas_ideal_constructor",nullptr,
    sizeof(ConstructorConfig),nullptr,nullptr,traverse_constructor,clear_constructor,nullptr};

// Dispatch ordinary declarations directly. Scalar embeddings, ordered results,
// worker-sized ideals and diagnostics continue through the public validator.
PyObject* public_ideal(PyObject* module, PyObject* args, PyObject* kwargs) {
    auto* state = static_cast<ConstructorConfig*>(PyModule_GetState(module))->definition;
    auto* base = reinterpret_cast<PyTypeObject*>(PyTuple_GetItem(state,0));
    auto* subclass = reinterpret_cast<PyTypeObject*>(PyTuple_GetItem(state,1));
    auto* polynomial_type = PyTuple_GetItem(state,2);
    auto fallback = [&] { return PyObject_Call(PyTuple_GetItem(state,3),args,kwargs); };
    auto* order = PyTuple_GetItem(state,4);
    PyObject* requested_ring = nullptr;
    if (kwargs) {
        requested_ring = PyDict_GetItemString(kwargs,"ring");
        auto* chosen = PyDict_GetItemString(kwargs,"order");
        if (PyDict_Size(kwargs) != (requested_ring != nullptr) + (chosen != nullptr)) return fallback();
        if (chosen && chosen != Py_None) {
            if (!PyUnicode_CheckExact(chosen) ||
                (PyUnicode_CompareWithASCIIString(chosen,"grevlex") && PyUnicode_CompareWithASCIIString(chosen,"lex")))
                return fallback();
            order = chosen;
        }
    }
    PyObject* items = args;
    if (PyTuple_Size(args) == 1 && PyTuple_CheckExact(PyTuple_GetItem(args,0))) items = PyTuple_GetItem(args,0);
    const auto count = PyTuple_Size(items);
    if (!count || count > 128) return fallback();
    const auto* fields = static_cast<const Fields*>(PyType_GetModuleState(base));
    if (!fields) return nullptr;
    OwnedReference parent{nullptr};
    void* ring = nullptr;
    void* polynomials[128];
    PolynomialOwners owners;
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* f = PyTuple_GetItem(items,i);
        if (reinterpret_cast<PyObject*>(Py_TYPE(f)) != polynomial_type) return fallback();
        auto* space = PyObject_GetAttr(f,fields->parent_name);
        if (!space) return nullptr;
        if (i == 0) {
            parent.value = space;
            if (requested_ring && requested_ring != Py_None && parent.value != requested_ring) return fallback();
        } else {
            bool same = space == parent.value;
            Py_DECREF(space);
            if (!same) return fallback();
        }
        auto* owner = PyObject_GetAttr(f,fields->polynomial_name);
        if (!owner) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError)) return nullptr;
            PyErr_Clear();
            return fallback();
        }
        owners.objects[owners.count++] = owner;
        if (!PyCapsule_IsValid(owner,polynomial_name)) return fallback();
        polynomials[i] = PyCapsule_GetPointer(owner,polynomial_name);
        if (i == 0) {
            auto* context = static_cast<PyObject*>(PyCapsule_GetContext(owner));
            ring = PyCapsule_GetPointer(context,ring_name);
            if (!ring) return nullptr;
        }
    }
    void* handle = atlas_ideal_from_polynomials(ring,polynomials,static_cast<size_t>(count));
    if (!handle && !*atlas_ideal_error()) return fallback();
    auto* result = owned(base,handle,subclass);
    if (result) {
        auto* declared = reinterpret_cast<Handle*>(result);
        declared->space = Py_NewRef(parent.value);
        declared->generators = Py_NewRef(items);
        declared->order = Py_NewRef(order);
    }
    return result;
}

PyMethodDef ideal_function = {"ideal", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)()>(public_ideal)),
    METH_VARARGS | METH_KEYWORDS,
    "ideal($self, /, *generators, order=None, variables=None, ring=None)\n--\n\n"
    "Declare an ideal in its generators' ring or choose one with ring=f.parent or ring=QQ."};

PyObject* make_constructor(PyObject* type, PyObject* args) {
    PyObject *value_type, *polynomial_type, *fallback;
    if (!PyArg_ParseTuple(args,"O!O!O",&PyType_Type,&value_type,&PyType_Type,&polynomial_type,&fallback))
        return nullptr;
    if (!PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(value_type),reinterpret_cast<PyTypeObject*>(type))) {
        PyErr_SetString(PyExc_TypeError,"Expected a native ideal value type.");
        return nullptr;
    }
    OwnedReference order{PyUnicode_InternFromString("grevlex")};
    if (!order.value) return nullptr;
    // A module-bound builtin retains its qualified name, signature and pickle identity.
    OwnedReference context{PyModule_Create(&constructor_module)};
    if (!context.value) return nullptr;
    auto* config = static_cast<ConstructorConfig*>(PyModule_GetState(context.value));
    config->definition = PyTuple_Pack(5,type,value_type,polynomial_type,fallback,order.value);
    OwnedReference module{PyObject_GetAttrString(fallback,"__module__")};
    if (!config->definition || !module.value) return nullptr;
    return PyCFunction_NewEx(&ideal_function,context.value,module.value);
}

PyObject* copy_handle(PyObject* self, PyObject*) { return Py_NewRef(self); }
PyObject* native_ideal(PyObject* self, void*) { return Py_NewRef(self); }
PyMethodDef methods[] = {
    {"prepare", prepare, METH_CLASS | METH_VARARGS, nullptr},
    {"ring", make_ring, METH_CLASS | METH_O, nullptr},
    {"polynomial", make_polynomial, METH_CLASS | METH_VARARGS, nullptr},
    {"polynomial_gcd", polynomial_gcd, METH_CLASS | METH_VARARGS, nullptr},
    {"polynomial_is_zero", polynomial_is_zero, METH_CLASS | METH_O, nullptr},
    {"polynomial_row", polynomial_row, METH_CLASS | METH_O, nullptr},
    {"from_polynomials", from_polynomials, METH_CLASS | METH_VARARGS, nullptr},
    {"constructor", make_constructor, METH_CLASS | METH_VARARGS, nullptr},
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
    {"_space", T_OBJECT_EX, offsetof(Handle,space), READONLY, nullptr},
    {"generators", T_OBJECT_EX, offsetof(Handle,generators), READONLY, nullptr},
    {"order", T_OBJECT_EX, offsetof(Handle,order), READONLY, nullptr},
    {nullptr,0,0,0,nullptr}
};
PyGetSetDef getters[] = {
    {"_native_ideal", native_ideal, nullptr, nullptr, nullptr},
    {nullptr,nullptr,nullptr,nullptr,nullptr}
};
PyType_Slot slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(reject_new)},
    {Py_tp_dealloc, reinterpret_cast<void*>(destroy)},
    {Py_tp_traverse, reinterpret_cast<void*>(traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(clear)},
    {Py_tp_members, members}, {Py_tp_methods, methods}, {Py_tp_getset, getters}, {0,nullptr}
};
PyType_Spec spec = {"_atlas_ideal.Ideal",sizeof(Handle),0,
                    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE,slots};

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
    if (atlas_add_session_runner(m) < 0) { Py_DECREF(m); return nullptr; }
    auto* fields = static_cast<Fields*>(PyModule_GetState(m));
    fields->parent_name = PyUnicode_InternFromString("_space");
    fields->polynomial_name = PyUnicode_InternFromString("_native_polynomial");
    if (!fields->parent_name || !fields->polynomial_name) { Py_DECREF(m); return nullptr; }
    auto* type = PyType_FromModuleAndSpec(m,&spec,nullptr);
    if (!type || PyModule_AddObject(m,"Ideal",type) < 0) { Py_XDECREF(type); Py_DECREF(m); return nullptr; }
    return m;
}
