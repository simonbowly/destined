
import itertools
import operator


def evaluate_distribution(spec, function_lookup):
    ''' Process the declarative specification and return a function
    of the form:

        def wrapper(rstate, **kwargs):
            ...

    Regardless of the specification, the generated function expects a
    positional argument which is a random number generator (like
    np.random.RandomState), and a set of keyword arguments.
    '''

    if 'value' in spec:

        assert len(spec) == 1
        _wrapped_value = spec['value']

        def wrapper(rstate, **kwargs):
            return _wrapped_value

        wrapper._exposed_kwargs = set()
        return wrapper

    elif 'function' in spec or 'generator' in spec:

        assert set(spec.keys()).issubset({
            'function', 'generator', 'parameters', 'declare'})

        if 'function' in spec:
            assert 'generator' not in spec
            _wrapped_function = function_lookup(spec['function'])
            def _wrapped_generator(rstate, **kwargs):
                return _wrapped_function(**kwargs)
        else:
            _wrapped_generator = function_lookup(spec['generator'])

        exposed_kwargs_map = dict()
        param_callables = dict()
        if 'parameters' in spec:
            param_callables = {
                param_name: evaluate_distribution(
                    param_spec, function_lookup)
                for param_name, param_spec in spec['parameters'].items()
                if 'expose' not in param_spec
            }
            exposed_kwargs_map = {
                param_name: param_spec['expose']
                for param_name, param_spec in spec['parameters'].items()
                if 'expose' in param_spec
            }

        _exposed_kwargs = set(exposed_kwargs_map.values())
        _exposed_kwargs.update(itertools.chain(*(
            param_callable._exposed_kwargs
            for param_callable in param_callables.values()
            )))

        declared_callables = dict()
        if 'declare' in spec:
            declared_callables = {
                declared_name: evaluate_distribution(
                    declared_spec, function_lookup)
                for declared_name, declared_spec
                in spec['declare'].items()
            }

        _exposed_kwargs.update(itertools.chain(*(
            declared_callable._exposed_kwargs
            for declared_callable in declared_callables.values()
            )))
        _exposed_kwargs = {
            kwarg
            for kwarg in _exposed_kwargs
            if kwarg not in declared_callables
        }

        def wrapper(rstate, **kwargs):
            missing_kwargs = set(_exposed_kwargs) - set(kwargs)
            if missing_kwargs:
                str_missed = ', '.join('\'{}\''.format(kw) for kw in sorted(missing_kwargs))
                raise TypeError('function missing required keyword-only arguments: {}'.format(str_missed))
            inner_kwargs = {
                inner_kw: kwargs[outer_kw]
                for inner_kw, outer_kw in exposed_kwargs_map.items()
            }

            kwargs.update({
                declared_name: declared_callable(rstate, **kwargs)
                for declared_name, declared_callable
                in sorted(declared_callables.items(), key=operator.itemgetter(0))
                })

            inner_kwargs.update({
                param_name: param_callable(rstate, **kwargs)
                for param_name, param_callable
                in sorted(param_callables.items(), key=operator.itemgetter(0))
                })
            return _wrapped_generator(rstate, **inner_kwargs)

        wrapper._exposed_kwargs = _exposed_kwargs
        return wrapper

    else:
        raise ValueError(spec)
