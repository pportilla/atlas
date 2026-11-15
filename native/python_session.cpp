// Compile straight-line named assignments for the Atlas script evaluator.
// Only the syntax is retained. Each execution reads the current session values.
#define Py_LIMITED_API 0x030a0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <memory>
#include <new>
#include <vector>

namespace {
constexpr char program_type[] = "atlas.script_assignments";

struct Instruction {
    PyObject *target, *operation, *first, *second;
    bool literal;
};

struct Program {
    // The two exact tuples own the names and literals referenced by these pointers.
    PyObject *source, *keys;
    PyObject *variables_key, *builtins_key, *reserved_key, *coordinates_key;
    std::vector<Instruction> instructions;

    Program(PyObject* syntax, PyObject* names) : source(Py_NewRef(syntax)), keys(Py_NewRef(names)),
        variables_key(PyTuple_GetItem(keys, 0)), builtins_key(PyTuple_GetItem(keys, 1)),
        reserved_key(PyTuple_GetItem(keys, 2)), coordinates_key(PyTuple_GetItem(keys, 3)) {}
    ~Program() { Py_DECREF(source); Py_DECREF(keys); }
};

void release_program(PyObject* capsule) {
    delete static_cast<Program*>(PyCapsule_GetPointer(capsule, program_type));
}

// Keep the callable and both arguments alive if it changes its own bindings.
struct Arguments {
    PyObject *function, *left, *right;
    Arguments(PyObject* f, PyObject* a, PyObject* b) :
        function(Py_NewRef(f)), left(Py_NewRef(a)), right(Py_NewRef(b)) {}
    ~Arguments() { Py_DECREF(right); Py_DECREF(left); Py_DECREF(function); }
};

// Python supplies the source line and function name from the failing instruction.
PyObject* call_error(Py_ssize_t index) {
    if (!PyErr_ExceptionMatches(PyExc_Exception)) return nullptr;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    Py_XDECREF(type);
    if (value && traceback && PyException_SetTraceback(value, traceback) < 0) Py_CLEAR(value);
    Py_XDECREF(traceback);
    return value ? Py_BuildValue("(nN)", index, value) : nullptr;
}

PyObject* execute(PyObject* capsule, PyObject* session) {
    auto* program = static_cast<const Program*>(PyCapsule_GetPointer(capsule, program_type));
    if (!program) return nullptr;
    if (!PyDict_CheckExact(session)) return PyLong_FromLong(0);
    Py_ssize_t i = 0;
    for (const Instruction& call : program->instructions) {
        auto* variables = PyDict_GetItemWithError(session, program->variables_key);
        auto* builtins = PyDict_GetItemWithError(session, program->builtins_key);
        if (!variables || !PyDict_CheckExact(variables) || !builtins || !PyDict_CheckExact(builtins)) break;
        auto* left = PyDict_GetItemWithError(variables, call.first);
        auto* right = call.literal ? call.second : PyDict_GetItemWithError(variables, call.second);
        if (!left || !right) break;
        auto* function = PyDict_GetItemWithError(variables, call.operation);
        if (!function && !PyErr_Occurred()) function = PyDict_GetItemWithError(builtins, call.operation);
        // Coordinate declarations use the evaluator's additional binding checks.
        if (!function || !PyCallable_Check(function) ||
            function == PyDict_GetItemWithError(builtins, program->coordinates_key)) break;
        if (PyErr_Occurred()) return nullptr;
        Arguments args(function, left, right);
        PyObject* result = PyObject_CallFunctionObjArgs(args.function, args.left, args.right, nullptr);
        if (!result) return call_error(i);

        // An application-supplied callable can replace the workspace or change
        // the reserved names before its result is assigned.
        builtins = PyDict_GetItemWithError(session, program->builtins_key);
        auto* reserved = PyDict_GetItemWithError(session, program->reserved_key);
        variables = PyDict_GetItemWithError(session, program->variables_key);
        if (!builtins || !reserved || !variables) {
            Py_DECREF(result);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_AttributeError, "Missing Atlas session bindings.");
            return nullptr;
        }
        int blocked = PyDict_CheckExact(builtins) ? PyDict_Contains(builtins, call.target) :
                                                    PySequence_Contains(builtins, call.target);
        if (!blocked) blocked = PyFrozenSet_CheckExact(reserved) ? PySet_Contains(reserved, call.target) :
                                                                 PySequence_Contains(reserved, call.target);
        if (blocked) {
            Py_DECREF(result);
            return blocked < 0 ? nullptr : Py_BuildValue("(nO)", i, Py_None);
        }
        const int status = PyObject_SetItem(variables, call.target, result);
        Py_DECREF(result);
        if (status < 0) return nullptr;
        ++i;
    }
    // Resume at the first unsupported instruction, leaving earlier assignments in place.
    return PyErr_Occurred() ? nullptr : PyLong_FromSsize_t(i);
}

PyMethodDef execute_method = {"run_assignments", execute, METH_O, nullptr};

PyObject* compile_assignments(PyObject* keys, PyObject* syntax) {
    if (!PyTuple_CheckExact(syntax)) {
        PyErr_SetString(PyExc_TypeError, "Expected a tuple of Atlas assignments.");
        return nullptr;
    }
    std::unique_ptr<Program> program;
    try {
        program = std::make_unique<Program>(syntax, keys);
        const auto count = PyTuple_Size(syntax);
        program->instructions.reserve(static_cast<size_t>(count));
        for (Py_ssize_t i = 0; i < count; ++i) {
            auto* row = PyTuple_GetItem(syntax, i);
            if (!PyTuple_CheckExact(row) || PyTuple_Size(row) != 5) {
                PyErr_SetString(PyExc_ValueError, "Invalid Atlas assignment.");
                return nullptr;
            }
            auto* target = PyTuple_GetItem(row, 0);
            auto* operation = PyTuple_GetItem(row, 1);
            auto* first = PyTuple_GetItem(row, 2);
            auto* second = PyTuple_GetItem(row, 3);
            const bool literal = PyLong_CheckExact(second);
            if (!PyUnicode_CheckExact(target) || !PyUnicode_CheckExact(operation) ||
                !PyUnicode_CheckExact(first) || !(literal || PyUnicode_CheckExact(second)) ||
                !PyLong_CheckExact(PyTuple_GetItem(row, 4))) {
                PyErr_SetString(PyExc_TypeError, "Invalid names or literal in an Atlas assignment.");
                return nullptr;
            }
            program->instructions.push_back({target, operation, first, second, literal});
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    auto* capsule = PyCapsule_New(program.get(), program_type, release_program);
    if (!capsule) return nullptr;
    program.release();
    auto* callable = PyCFunction_New(&execute_method, capsule);
    Py_DECREF(capsule);
    return callable;
}

PyMethodDef compile_method = {"compile_assignments", compile_assignments, METH_O, nullptr};
}

int atlas_add_session_runner(PyObject* module) {
    auto* keys = Py_BuildValue("(ssss)", "variables", "builtins", "reserved", "coordinates");
    if (!keys) return -1;
    auto* compiler = PyCFunction_New(&compile_method, keys);
    Py_DECREF(keys);
    if (!compiler) return -1;
    const int status = PyModule_AddObject(module, "compile_assignments", compiler);
    if (status < 0) Py_DECREF(compiler);
    return status;
}
