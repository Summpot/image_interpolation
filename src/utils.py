import inspect


def get_public_functions(module):
    return [
        member
        for name, member in inspect.getmembers(module)
        if inspect.isroutine(member)
        and not name.startswith("_")
        and (
            getattr(member, "__module__") == module.__name__
            or inspect.isbuiltin(member)  # for pyo3
        )
    ]
