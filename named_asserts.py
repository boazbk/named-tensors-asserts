############################################
# Helper classes for keeping track of named tensor dimensions
# General idea:
# To say that a tensor T has dimensions  ['batch','width','height','channels']
# We write: T &nt // "batch width height channels"
#
# To say that a model maps ['batch','width','height','channels'] to ['batch','output']
# We we write: model &nt // "batch, width, height, channels -> batch, output"
# (Commas can be dropped when they are not needed)

# To update a dimension we can use an excalamation mark.
# For example, if T has shape (10,2,3) then T &nt // "!batch width height"
# will update batch and and assert that width and height are 2 and 3 respectively.
# We can also write T &nt // "batch =10, width, height"

# Dependencies: torch, einops

import torch
import re
import ast

try:
    import einops
    einops_available = True
    from einops import rearrange
except ImportError:
    einops_available = False
    print("Warning: einops not available, some functionality will be disabled")
    def rearrange(*args, **kwargs):
        raise ImportError("einops not available")

from contextlib import contextmanager


# From https://stackoverflow.com/a/47130538
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
        self.__dict__.update(dict(dims = {}, _enabled = True, use_named_tensors = False , bound_exp = None , dims_in = None, dims_out  = None, type_out = None))
    
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
    
    # def __call__(self,*args,**kwargs):
    #     """Return a tuple while also updating dimensions from keyword arugments"""
    #     D = {k:v for k,v in kwargs.items() if isinstance(k,str) and (k[0]!='_')}
    #     self.set(**D)
    #     return tuple(args + kwargs.values())

        
    def set(self,**vals):
        for k,v in vals.items():
            if not k.isidentifier():
                raise ValueError(f"Invalid dimension name {k}")
            if (k in self.dims) and (v != self.dims[k]):
                print("Warning: Overwriting  {} from {} to {}".format(k,self.dims[k],v))
            if not k in self.dims:
                print(f"Updating {k} to {v}")
            self.dims[k]=v

    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False

    def eval_dim(self,exp):
        original_exp = exp
        exp = '\n'.join(exp.split(';'))
        try:
            result = exec_return(exp, globals(),self.dims,allow_statement=False)
        except Exception as e:
            raise ValueError(f"Could not evaluate dimension expression: {original_exp}, error in {exp}: {e} [evaluated wrt {self.dims}]")
        return result
    
    #https://gist.github.com/rockt/a3191f517728ea9a136a204f578d27c8
    @staticmethod
    def einsumfy_exp(exp):
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


    def validate_exp(self,exp):
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
        L = self.split_list(exp1)+self.split_list(exp2)+self.split_list(exp3)
        for n in L:
            if (n[0]=='!'):
                if not n[1:].isidentifier(): 
                    raise ValueError(f"Dimension name not valid identifier: {n} in the expression {origin_exp}")
                else:
                    pass 
            else:
                try:
                    v = self.eval_dim(n)
                except Exception as e:
                    raise ValueError(f"Dimension expression not valid: {n} in the expression {origin_exp}: {e}")
            
            
            


    @staticmethod
    def split_list(exp):
        if not exp: return []
        exp = exp.replace("[","").replace("]","")
        
        if ',' in exp:
            L = exp.split(',')
        else:
            L = re.split("\s|(?<!\d)[,.](?!\d)",exp)
        L = [x.strip() for x in L]
        L = [x for x in L if x != '']
        return L

    def assert_integer(self,I_,name):
        try:
            I = int(I_)
        except:
            raise ValueError(f"{name} must be an integer, got {I}")
        name_ = name[1:] if name[0]=='!' else name
        if name[0]=='!':
                if name_.isidentifier():
                    self.set(**{name_:I})
                else:
                    raise AssertionError(f"Dimension name not valid identifier: {name_}")
        else:
            assert (val := self.eval_dim(name_)) == I, f"Dimension {name_} must be {I}, got {val}"
        return I_

            




    def assert_tensor_dims(self,T, names):
        """Assert tensor T has the dimensions specified by names"""
        if not self._enabled: return T
        assert len(names)==len(T.shape), "Expected {} dims, got {}  (shape={} / names = {})".format(len(names), len(T.shape), T.shape, names)
        new_names = list(T.names)
        for i,name in enumerate(names):
            name_ = name[1:] if name[0]=='!' else name
            if T.names[i] and name_.isidentifier():
                assert T.names[i] == name_, "Expected dim {} to be named {} but got {}  (shape={} / names = {})".format(
                    i, name_, T.names[i], T.shape, names)
            
            if name[0]=='!':
                if name_.isidentifier():
                    self.set(**{name_:T.shape[i]})
                else:
                    raise AssertionError(
                        "Invalid dimension name {}  (shape={} / names={})".format(name[1:],  T.shape, names))
            assert self.eval_dim(name_) == T.shape[i], "Expected dim {} to be len({})={} but got {} (shape={} / names={})".format(
                i, name, self.eval_dim(name_), T.shape[i], T.shape, names)
        if name_.isidentifier():
                new_names[i] = name_
        if self.use_named_tensors:
            return T.refine_names(*new_names)
        return T
    
    def assert_model_dims(self,model,names_in,names_out):
        if not self._enabled: return model
        dims = [self.eval_dim(i) for i in names_in]
        try:
            device = next(model.parameters()).device
            x = torch.randn(*dims, device = device)
        except Exception as e:
            print(f"Error in creating input on {device} while asserting model {model} wrt {names_in}->{names_out} ({e})")
            raise e
        self.assert_tensor_dims(x,names_in)
        try:
            y = model(x)
        except Exception as e:
            print(f"Error in evaluating {model} on tensor of {x.shape} (device= {device}) while asserting model {model} wrt {names_in}->{names_out} ({e})")
            raise e
        self.assert_tensor_dims(y,names_out)
        return model
    

    
    def rearrange(self,T, exp):
        in_dims = self.split_list(exp[:exp.index('->')])
        out_dims = self.split_list(exp[exp.index('->')+2:])
        T = self.assert_tensor_dims(T, *in_dims)
        T =rearrange(T, exp)
        T =self.assert_tensor_dims(T, *out_dims)
        return T
    
    def einsum(self,exp,*args):
        out_dims = None
        if exp.index(": ")>0:
            out_dims = self.split_list(exp[exp.index(": ")+2:])
            exp = exp[:exp.index(": ")]
        exps_in = exp[:exp.index('->')].split(',')
        assert len(exps_in) == len(args), "Expected {} args, got {}".format(len(exps_in), len(args))
        for i,T in enumerate(args):
            self.assert_tensor_dims(T, self.split_list(exps_in[i]))
        out = torch.einsum(self.einsumfy_exp(exp), *args)
        if out_dims:
            self.assert_tensor_dims(out, out_dims)
        return out 
            
    def parse_exp(self,exp):
        dims_in = None
        dims_out = None
        if '->' in exp:
            dims_in = self.split_list(exp[:exp.index('->')])
            if ': ' in exp:
                dims_out = self.split_list(exp[exp.index(': ')+2:])
            else:
                dims_out = self.split_list(exp[exp.index('->')+2:])
        else:
            dims_in = self.split_list(exp)
        return dims_in, dims_out

    

    def __floordiv__(self,other):
        if isinstance(other, str):
            self.validate_exp(other)
            result = BoundNTOp()
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


    def __rand__(self,other):
        
        if not self.nt._enabled: return other
        if self.mode == "tensor":
            if isinstance(other, int):
                assert len(self.dims_in)==1, "Expected 1 dim, got {} (dims_in={})".format(len(self.dims_in), self.dims_in)
                return self.nt.assert_integer(other, self.dims_in[0])
            return self.nt.assert_tensor_dims(other, self.dims_in)
        if self.mode == "model":
            return self.nt.assert_model_dims(other, self.dims_in, self.dims_out)
        if self.mode == "einops":
            return self.nt.einops(self.bound_exp, other)
        if self.mode == "rearrange":
            return self.nt.rearrange(other, self.bound_exp)
        raise Exception("Unknown mode {}".format(self.mode))

nt= NamedTensorOp.instance()

            

@contextmanager
def skip_assert(debug = False):
    try:
        old_debug = NamedTensorOp.instance()._enabled
        NamedTensorOp.instance()._enabled = debug
        yield
    finally:
        NamedTensorOp.instance()._enabled = old_debug

