############################################
# Helper classes for making assertions about named tensor dimensions
# General idea:
# We write: T &nt // "batch=1024, channels=3, height=32, width=32" 
# to assert that T's shape is (1024,3,32,32) and to update the global
# named dimensions batch, channels,height, width to these values.

# We can access these with nt.batch, nt.width etc
# In future declarations we can write expressions such as: Q &nt // "batch, channels*(height+1), width"

# To say that a model maps ['batch','width','height','channels'] to ['batch','output']
# We we write: model &nt // "batch, width, height, channels -> batch, output"


# Dependencies: torch, einops (optional)

import torch
import re
import inspect
import sys
from contextlib import contextmanager

def _trunc(s, maxlen=30):
    if isinstance(s, torch.Tensor):
        s = str(s)[6:]
    else:
        s = str(s)
    if len(s) > maxlen:
        return s[:maxlen-2] + '..'
    return s

# From https://stackoverflow.com/a/47130538
# Right now not used, but might be used in the future
import ast
def exec_return(script, globals=None, locals=None, allow_statement=True):
    '''Execute a script and return the value of the last expression
    Modification: if last line contains :=, we add parenthesis around it
    if last line is an assignment, we make it into assignment expression'''
    lines = script.split('\n')
    if ':=' in lines[-1]:
        lines[-1] = lines[-1].strip()
        if lines[-1][0] != '(':
            lines[-1] = '(' + lines[-1] + ')'
    script = '\n'.join(lines)
    stmts = list(ast.iter_child_nodes(ast.parse(script)))
    if not stmts:
        return None
    if isinstance(stmts[-1],ast.Assign):
        lineno = stmts[-1].lineno
        col_offset = stmts[-1].col_offset
        end_lineno = stmts[-1].end_lineno
        end_col_offset = stmts[-1].end_col_offset
        exp = ast.NamedExpr(
            target=stmts[-1].targets[0], value=stmts[-1].value)
        exp.lineno = lineno
        exp.col_offset = col_offset
        exp.end_lineno = end_lineno
        exp.end_col_offset = end_col_offset
        stmts[-1] = ast.Expr(exp)
        stmts[-1].lineno = lineno
        stmts[-1].col_offset = col_offset
        stmts[-1].end_lineno = end_lineno
        stmts[-1].end_col_offset = end_col_offset

    if isinstance(stmts[-1], ast.Expr):
        # the last one is an expression and we will try to return the results
        # so we first execute the previous statements
        if len(stmts) > 1:
            exec(compile(ast.Module(body=stmts[:-1]), filename="<ast>", mode="exec"), globals, locals)
        # then we eval the last one
        return eval(compile(ast.Expression(body=stmts[-1].value), filename="<ast>", mode="eval"), globals, locals)
    else:
        if not allow_statement:
            raise SyntaxError(f"Last statement in {script} is not an expression: {ast.dump(stmts[-1])}")
        # otherwise we just execute the entire code
        return exec(script, globals, locals)


# From https://stackoverflow.com/a/7346105
class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

    
@Singleton
class NamedTensorOp:
    '''Singleton class that is meant to capture all named tensor dimensions, and parse comments accordingly
    If nt is the singleton element and S is a string corresponding to the dimension annotation, 
    then nt // S is a BoundNTOp object that corresponds to this annotation
    '''

    def __init__(self):
        self.__dict__.update(dict(dims = {}, _enabled = True,  bound_exp = None , dims_in = None, dims_out  = None, type_out = None, no_logs = False))
    
    def __setattr__(self, __name, __value):
        if __name in self.__dict__:
            self.__dict__[__name] = __value
        else:
            self.dims[__name] = __value
        
    
    def __getattr__(self,name):
        if name in self.dims:
            return self.dims[name]
        else:
            raise AttributeError(f"No dimension {name}")
    
    def __call__(self,*args,**kwargs):
        """Return a tuple while also updating dimensions from keyword arugments"""
        D = {k:v for k,v in kwargs.items() if isinstance(k,str) and (k[0]!='_')}
        self.set(**D)
        if len(kwargs.values()):
            return args + tuple(kwargs.values())
        return args 

    def log(self,msg):
        if not self.no_logs: print(msg, file=sys.stderr)

        
    def set(self,**vals):
        for k,v in vals.items():
            if not k.isidentifier():
                raise ValueError(f"Invalid dimension name {k}")
            if (k in self.dims) and (v != self.dims[k]):
                _log("Warning: Overwriting  {} from {} to {}".format(k,self.dims[k],v))
            if not k in self.dims:
                _log(f"Updating {k} to {v}")
            self.dims[k]=v

    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False

    def eval_dims(self,exp,context={}):
        try:
            result = eval(
                f"NamedTensorOp.instance()({exp})", {**globals(),**context}, self.dims)
        except Exception as e:
            raise ValueError(f"Could not evaluate dimension expression: {exp}: {e} [evaluated wrt {self.dims},  {context}]")
        return result
    
    #https://gist.github.com/rockt/a3191f517728ea9a136a204f578d27c8
    @staticmethod
    def einsumfy_exp(exp):
        exp = exp.replace(',',' ')
        names = set(re.split("[, \(\)]|->", exp))
        names.remove("")

        invalid_names = set(filter(lambda x: len(x) > 1, names))
        if "..." in invalid_names:
            invalid_names.remove("...")

        free_chr = ord('a')
        for name in invalid_names:
            while chr(free_chr) in names:
                free_chr += 1
            exp = exp.replace(name, chr(free_chr))
            free_chr += 1

        return exp


    def validate_exp(self,exp, context={}):
        origin_exp = exp
        if '-->' in exp:
            raise ValueError("--> is not allowed in the expression {exp}, use -> instead")
        exp1 = ""
        exp2 = ""
        exp3 = ""
        if ': ' in exp:
            exp3 = exp[exp.index(': ')+2:]
            exp = exp[:exp.index(': ')]
        if '->' in exp:
            exp2 = exp[exp.index('->')+2:]
            exp = exp[:exp.index('->')]
        exp1 = exp
        for e in [exp1,exp2,exp3]:
            e = self.preprocess_exp(e)
            if e:
                try:
                    self.eval_dims(e,context)
                except Exception as exception:
                    raise ValueError(f"Dimension expression not valid: {e} in the expression {origin_exp}: {exception}")
            
            
            


    @staticmethod
    def preprocess_exp(exp):
        if not isinstance(exp,str):
            raise ValueError(f"Expected string, got {exp} of type {type(exp)}")
        exp = exp.strip()
        if not exp: return exp
        if exp[0]=='(' and exp[-1]==')':
            exp = exp[1:-1]
        if exp[0]=='[' and exp[-1]==']':
            exp = exp[1:-1]
        return exp

    def assert_integer(self,I_,exp,context={}):
        try:
            I = int(I_)
        except:
            raise ValueError(f"{exp} must be an integer, got {I}")
        val = self.eval_dims(exp,context)
        assert len(val) == 1, f"{exp} must be a single integer, got {val}"
        assert val[0] == I, f"Dimension {exp} must be {I}, got {val[0]}"
        return I_

            
    



    def assert_tensor_dims(self,T, exp, context={}):
        """Assert tensor T has the dimensions specified by exp"""
        if not self._enabled: return T
        try:
            shape = self.eval_dims(exp,context)
        except Exception as e:
            raise ValueError(f"Dimension expression not valid: {exp}: {e}")
        assert len(shape)==len(T.shape), f"Dimension expression {exp} must have the same number of dimensions as the tensor {_trunc(T)}, got {shape} vs {T.shape}"
        assert shape == T.shape, f"Dimension expression {exp} must have the same dimensions as the tensor {_trunc(T)}, got {shape} vs {T.shape}"
        return T
    
    def assert_model_dims(self,model,exp_in,exp_out, context={}):
        if not self._enabled: return model
        try:
            shape_in = self.eval_dims(exp_in,context)
        except Exception as e:
            raise ValueError(f"Input dimension expression not valid: {exp_in}: {e}")
        try:
            shape_out = self.eval_dims(exp_out,context)
        except Exception as e:
            raise ValueError(f"Output dimension expression not valid: {exp_out}: {e}")
        device = torch.device("cpu")
        try:
            device = next(model.parameters()).device
        except Exception as e:
            pass
        try:
            x = torch.randn(*shape_in, device = device)
        except Exception as e:
            _log(
                f"Error in creating input on {device} while asserting model {model} wrt {shape_in}->{shape_out} ({e}, {exp_in}->{exp_out})")
            raise e
        try:
            self.assert_tensor_dims(x,exp_in, context=context)
        except Exception as e:
            _log(f"Error in asserting input dimension {shape_in} (expression {exp_in}->{exp_out}), input shape {x.shape}")
            raise e

        try:
            y = model(x)
        except Exception as e:
            _log(f"Error in evaluating {model} on tensor of {x.shape} (device= {device}) while asserting model {model} wrt {shape_in}->{shape_out} ({e}, {exp_in}->{exp_out})")
            raise e
        try:
            self.assert_tensor_dims(y, exp_out, context=context)
        except Exception as e:
            _log(f"Error in asserting output dimension {shape_out} (expression {exp_in}->{exp_out}), output shape {y.shape}")
            raise e
        return model
    

    
    def rearrange(self,T, exp, context= {}):
        in_dims = self.preprocess_exp(exp[:exp.index('->')])
        out_dims = self.preprocess_exp(exp[exp.index('->')+2:])
        T = self.assert_tensor_dims(T, *in_dims, context= context)
        T =rearrange(T, exp)
        T =self.assert_tensor_dims(T, *out_dims, context= context)
        return T
    
    def einsum(self,exp,*args, context={}):
        out_dims = None
        if exp.index(": ")>0:
            out_dims = self.preprocess_exp(exp[exp.index(": ")+2:])
            exp = exp[:exp.index(": ")]
        exps_in = exp[:exp.index('->')].split(',')
        assert len(exps_in) == len(args), "Expected {} args, got {}".format(len(exps_in), len(args))
        for i,T in enumerate(args):
            self.assert_tensor_dims(T, self.preprocess_exp(exps_in[i]), context= context)
        out = torch.einsum(self.einsumfy_exp(exp), *args)
        if out_dims:
            self.assert_tensor_dims(out, out_dims, context= context)
        return out 
            
    def parse_exp(self,exp):
        dims_in = None
        dims_out = None
        if '->' in exp:
            dims_in = self.preprocess_exp(exp[:exp.index('->')])
            if ': ' in exp:
                dims_out = self.preprocess_exp(exp[exp.index(': ')+2:])
            else:
                dims_out = self.preprocess_exp(exp[exp.index('->')+2:])
        else:
            dims_in = self.preprocess_exp(exp)
        return dims_in, dims_out

    

    def __floordiv__(self,other):
        if isinstance(other,list) or isinstance(other, tuple):
            if not all(isinstance(x,int) for x in other):
                raise ValueError(f"Expected list of integers, got {other}")
            _log(f'Warning: expression is a list or tuple, interpreting as "{str(other)}"')
            other = str(other)
        if isinstance(other, str):
            result = BoundNTOp()
            context = {}
            try:
                frame = inspect.currentframe().f_back
                context = dict(frame.f_locals)
            except Exception as e:
                _log(f"Couldn't get local variables {e}")
            result.context = context 
            self.validate_exp(other, context=context)
            result.bound_exp = other
            result.dims_in, result.dims_out = self.parse_exp(other)
            if result.dims_out is None:
                result.mode = "tensor"
            else:
                result.mode = "model"
            return result
        else:
            return NotImplemented




class BoundNTOp:
    '''Object representation a dimension annotation. If X is a BoundNTOp object, and T is a model or tensor, 
    then T & X will raise an assertion error if X doesn't match T's expression '''

    def __init__(self):
        self.mode = None # One of [None, "tensor", "model", "rearrange", "einops"]
        self.nt = NamedTensorOp.instance()
        self.bound_exp = None
        self.dims_in = None
        self.dims_out = None
        self.context = {}


    def __rand__(self,other):
        
        if not self.nt._enabled: return other
        if self.mode == "tensor":
            if isinstance(other, int):
                return self.nt.assert_integer(other, self.dims_in, context=self.context)
            return self.nt.assert_tensor_dims(other, self.dims_in , context=self.context)
        if self.mode == "model":
            return self.nt.assert_model_dims(other, self.dims_in, self.dims_out, context=self.context)
        if self.mode == "einops":
            return self.nt.einsum(self.bound_exp, other, context=self.context)
        if self.mode == "rearrange":
            return self.nt.rearrange(other, self.bound_exp, context=self.context)
        raise Exception("Unknown mode {}".format(self.mode))

@contextmanager
def skip_assert(skip = True):
    try:
        old_debug = NamedTensorOp.instance()._enabled
        NamedTensorOp.instance()._enabled = not skip
        yield
    finally:
        NamedTensorOp.instance()._enabled = old_debug


nt= NamedTensorOp.instance()

def _log(msg):
    NamedTensorOp.instance().log(msg)

try:
    import einops
    einops_available = True
    from einops import rearrange
except ImportError:
    einops_available = False
    _log("Warning: einops not available, some functionality will be disabled")
    def rearrange(*args, **kwargs):
        raise ImportError("einops not available")

            

