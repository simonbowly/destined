
# TODO tests should verify that nothing is called before
# the generated function is executed. Either check not_called
# before executing func() and/or repeat the result check after
# resetting all the mocks.

from unittest.mock import Mock

import pytest

from destined import evaluate_distribution


def test_value():
    ''' Only one use case here. All arguments ignored and
    a constant is returned. '''
    spec = {'value': 'result'}
    func = evaluate_distribution(spec, None)
    del spec['value']
    assert func._exposed_kwargs == set()
    assert func('rstate') == 'result'
    assert func('rstate', a=1, b=1) == 'result'


def test_noarg_function():
    ''' function is just a special case of generator, where
    rstate is ignored. This is implemented using a wrapper
    function, so further tests can just be performed for the
    generator spec. '''
    target_func = Mock()
    function_lookup = {'return': target_func}
    spec = {'function': 'return'}
    func = evaluate_distribution(spec, function_lookup.__getitem__)
    del function_lookup['return']
    del spec['function']
    assert func._exposed_kwargs == set()
    result = func('rstate')
    target_func.assert_called_once_with()
    assert result == target_func()


def test_generator_noarg():
    target_gen = Mock()
    generator_lookup = {'return': target_gen}
    spec = {'generator': 'return'}
    func = evaluate_distribution(spec, generator_lookup.__getitem__)
    del generator_lookup['return']
    del spec['generator']
    assert func._exposed_kwargs == set()
    result = func('rstate')
    target_gen.assert_called_once_with('rstate')
    assert result == target_gen()


def test_generator_expose():
    target_gen = Mock()
    generator_lookup = {'return': target_gen}
    spec = {
        'generator': 'return',
        'parameters': {
            'a': {'expose': 'b'}
        }
    }
    func = evaluate_distribution(spec, generator_lookup.__getitem__)
    del generator_lookup['return']
    del spec['generator']
    del spec['parameters']
    assert func._exposed_kwargs == {'b'}
    result = func('rstate', b=2)
    target_gen.assert_called_once_with('rstate', a=2)
    assert result == target_gen()
    with pytest.raises(TypeError):
        func('rstate')


def test_generator_recursive():
    target_gen = Mock()
    generator_lookup = {'return': target_gen}
    spec = {
        'generator': 'return',
        'parameters': {
            'a': {'value': 1}
        }
    }
    func = evaluate_distribution(spec, generator_lookup.__getitem__)
    del generator_lookup['return']
    del spec['generator']
    del spec['parameters']
    assert func._exposed_kwargs == set()
    result = func('rstate')
    target_gen.assert_called_once_with('rstate', a=1)
    assert result == target_gen()


def test_recursive_passed():
    target_f1 = Mock()
    target_g1 = Mock()
    lookup = {
        'f1': target_f1,
        'g1': target_g1}
    spec = {
        'function': 'f1',
        'parameters': {
            'a': {'value': 2},
            'b': {
                'generator': 'g1',
                'parameters': {
                    'c': {'expose': 'outer_c'}
                }
            }
        }
    }
    func = evaluate_distribution(spec, lookup.__getitem__)
    del lookup['f1']
    del lookup['g1']
    del spec['function']
    del spec['parameters']
    assert func._exposed_kwargs == {'outer_c'}
    result = func('rstate', outer_c=1)
    target_g1.assert_called_once_with('rstate', c=1)
    target_f1.assert_called_once_with(a=2, b=target_g1())
    assert result == target_f1()
    with pytest.raises(TypeError):
        func('rstate')


def test_declare():
    target_f1 = Mock()
    target_f2 = Mock()
    target_f3 = Mock()
    function_lookup = {
        'f1': target_f1,
        'f2': target_f2,
        'f3': target_f3
    }
    spec = {
        'function': 'f1',
        'parameters': {
            'a': {
                'function': 'f2',
                'parameters': {
                    'b': {'value': 1},
                    'c': {'expose': 'arg1'}
                }
            }
        },
        'declare': {
            'arg1': {
                'function': 'f3',
                'parameters': {
                    'd': {'expose': 'arg2'}
                }
            }
        }
    }
    func = evaluate_distribution(spec, function_lookup.__getitem__)
    del function_lookup['f1']
    del function_lookup['f2']
    del function_lookup['f3']
    assert func._exposed_kwargs == {'arg2'}

    result = func('rstate', arg2=2)
    target_f3.assert_called_once_with(d=2)
    target_f2.assert_called_once_with(b=1, c=target_f3())
    target_f1.assert_called_once_with(a=target_f2())
    assert result == target_f1()

    target_f1.reset_mock()
    target_f2.reset_mock()
    target_f3.reset_mock()

    # declarations override any input arguments
    result = func('rstate', arg1=4, arg2=2)
    target_f3.assert_called_once_with(d=2)
    target_f2.assert_called_once_with(b=1, c=target_f3())
    target_f1.assert_called_once_with(a=target_f2())
    assert result == target_f1()



def test_eval_order():
    ''' Tests the order of evaluation w.r.t. the random number generator,
    which directly impacts the result.
    The defined order of evaluation at each level is:
        1. declarations, in sorted name order
        2. parameters, in sorted name order
        3. target function
    '''
    target_g1 = Mock(side_effect=lambda r: r())
    def mock_func(rstate, **kwargs):
        return rstate()
    def returning(arg):
        return arg
    target_g2 = Mock(side_effect=mock_func)
    lookup = {
        'return': returning,
        'g1': target_g1,
        'g2': target_g2
    }
    spec = {
        'generator': 'g2',
        'declare': {
            'arg1': {'generator': 'g1'},
            'arg2': {'generator': 'g1'}
        },
        'parameters': {
            'a': {'generator': 'g1'},
            'b': {'generator': 'g1'},
            'c': {'generator': 'g1'},
            'd': {
                'function': 'return',
                'parameters': {
                    'arg': {'expose': 'arg1'}
                }
            },
            'e': {
                'function': 'return',
                'parameters': {
                    'arg': {'expose': 'arg2'}
                }
            }
        }
    }

    func = evaluate_distribution(spec, lookup.__getitem__)
    rstate = Mock(side_effect=[1, 2, 3, 4, 5, 6])
    result = func(rstate)
    target_g2.assert_called_once_with(rstate, d=1, e=2, a=3, b=4, c=5)
    assert result == 6
