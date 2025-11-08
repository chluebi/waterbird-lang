use std::collections::{HashMap};
use std::hash::{Hash, Hasher};
use std::fmt;
use slab::Slab;

use crate::ast::TypeLiteral;
use crate::{ast};

pub type Ptr = usize;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Value {
    Void,
    Int(i64),
    Bool(bool),
    String(Ptr),
    Tuple(Vec<Value>),
    List(Ptr),
    Dictionary(Ptr),
    Lambda(Ptr),
    FunctionPtr(String),
    NameSpacePtr(String)
}

impl Hash for Value {
    
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        match self {
            Value::Int(i) => i.hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Tuple(values) => values.iter().map(|v| v.hash(state)).collect(),
            Value::String(ptr) => ptr.hash(state), // thanks to string interning!
            _ => unreachable!()
        }
    }

}

impl Value {

    pub fn hashable(&self) -> bool {
        match self {
            Value::Int(_) |
            Value::Bool(_) |
            Value::String(_) => true,
            Value::Tuple(values) => values.iter().all(Value::hashable),
            _ => false
        }
    }

    pub fn get_type_name(&self) -> &'static str {
        match self {
            Value::Void => "void",
            Value::Int(_) => "int",
            Value::Bool(_) => "bool",
            Value::String(_) => "str",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dictionary(_) => "dict",
            Value::Lambda(_) => "lambda",
            Value::FunctionPtr(_) => "function",
            Value::NameSpacePtr(_) => "namespace"
        }
    }

    pub fn get_type_namespace(&self, loc: &ast::Loc) -> Result<String, InterpreterErrorMessage> {
        match self {
            Value::Int(_) => Ok(String::from("Int")),
            Value::Bool(_) => Ok(String::from("Bool")),
            Value::String(_) => Ok(String::from("String")),
            Value::Tuple(_) => Ok(String::from("Tuple")),
            Value::List(_) => Ok(String::from("List")),
            Value::Dictionary(_) => Ok(String::from("Dict")),
            _ => Err(InterpreterErrorMessage {error: InterpreterError::InvalidNamespace {}, loc: Some(loc.clone())})
        }
    }

}

#[derive(Debug, Clone)]
pub enum HeapObject {
    Str(String),
    List(Vec<Value>),
    Dictionary(HashMap<Value,Value>),
    Lambda {
       arguments: Vec<ast::LambdaArgument>,
       expr: Box<ast::LocExpr>
    }
}

#[derive(Debug)]
pub struct Heap {
    pub objects: Slab<HeapObject>,
    intern_map: HashMap<String, Ptr>,
    intern_ptr_counts: HashMap<Ptr, u64>,
}

impl Heap {
    pub fn new() -> Self {
        Heap { objects: Slab::new(), intern_map: HashMap::new(), intern_ptr_counts: HashMap::new() }
    }

    pub fn alloc(&mut self, object: HeapObject) -> Ptr {
        self.objects.insert(object)
    }

    pub fn get(&self, ptr: Ptr) -> Option<&HeapObject> {
        self.objects.get(ptr)
    }

    pub fn get_mut(&mut self, ptr: Ptr) -> Option<&mut HeapObject> {
        self.objects.get_mut(ptr)
    }

    pub fn free(&mut self, ptr: Ptr) {

        if let Some(count) = self.intern_ptr_counts.get_mut(&ptr) {
            *count -= 1;

            if *count == 0 {
                self.intern_ptr_counts.remove(&ptr); 

                let object = self.objects.remove(ptr);

                if let HeapObject::Str(s) = object {
                    self.intern_map.remove(&s);
                } else {
                    unreachable!("Non-string object found in intern_ptr_counts map for Ptr {}", ptr);
                }
            }
        } else if self.objects.contains(ptr) {
            self.objects.remove(ptr);
        }
    }

   pub fn intern_string(&mut self, s: String) -> Ptr {
        if let Some(&ptr) = self.intern_map.get(&s) {
            *self.intern_ptr_counts.get_mut(&ptr).unwrap() += 1;
            return ptr;
        }
        let ptr = self.objects.insert(HeapObject::Str(s.clone()));
        self.intern_map.insert(s, ptr);
        self.intern_ptr_counts.insert(ptr, 1);

        ptr
    }
}

pub struct DisplayValue<'a> {
    value: &'a Value,
    heap: &'a Heap,
    is_contained: bool, 
}

impl<'a> DisplayValue<'a> {
    pub fn new(value: &'a Value, heap: &'a Heap) -> DisplayValue<'a> {
        DisplayValue { value, heap, is_contained: false }
    }

    pub fn contained_display(&self, value: &'a Value) -> DisplayValue<'a> {
        DisplayValue { 
            value, 
            heap: self.heap, 
            is_contained: true 
        }
    }
}

impl<'a> fmt::Display for DisplayValue<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            Value::Void => write!(f, "()"),
            Value::Int(i) => write!(f, "{}", i),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(ptr) => {
                match self.heap.get(*ptr) {
                    Some(HeapObject::Str(s)) => {
                        if self.is_contained {
                            write!(f, "\"{}\"", s) 
                        } else {
                            write!(f, "{}", s)
                        }
                    }
                    _ => write!(f, "!!InvalidStrPtr({})!!", ptr),
                }
            }
            Value::Tuple(values) => {
                let s = values.iter()
                    .map(|v| format!("{}", self.contained_display(v))) 
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "({})", s)
            }
            Value::List(ptr) => {
                 match self.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => {
                        let s = l.iter()
                            .map(|v| format!("{}", self.contained_display(v)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        write!(f, "[{}]", s)
                    }
                    _ => write!(f, "!!InvalidListPtr({})!!", ptr),
                }
            }
            Value::Dictionary(ptr) => {
                match self.heap.get(*ptr) {
                    Some(HeapObject::Dictionary(d)) => {
                        let s = d.iter()
                            .map(|(k, v)| format!("{}: {}", 
                                self.contained_display(k), 
                                self.contained_display(v)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        write!(f, "{{{}}}", s)
                    }
                    _ => write!(f, "!!InvalidDictPtr({})!!", ptr),
                }
            }
            Value::Lambda(_) => write!(f, "<lambda>"),
            Value::FunctionPtr(name) => write!(f, "<function {}>", name),
            Value::NameSpacePtr(name) => write!(f, "<namespace {}>", name),
        }
    }
}

#[derive(Debug)]
pub struct Stack {
    pub frames: Vec<Vec<HashMap<String,Value>>>
}

impl Stack {
    pub fn new() -> Self {
        Stack { frames: vec![vec![HashMap::new()]] }
    }

    pub fn update_variable(&mut self, name: &str, value: Value) -> Option<Value> {

        // this defines the semantics of updating variables
        // we first check if the variable has been "declared" previously, this means we need to update it in that place
        for frame in self.frames.last_mut().unwrap().into_iter().rev() {
            if frame.contains_key(name) {
                return frame.insert(String::from(name), value);
            }
        }

        // if not, this is the first declaration
        let newest_frame = self.frames.last_mut().unwrap().last_mut().unwrap();
        newest_frame.insert(String::from(name), value)
    } 

    pub fn contains_variable(&mut self, name: &str) -> bool {
        for frame in self.frames.last_mut().unwrap().iter().rev() {
            if frame.contains_key(name) {
                return true;
            }
        }
        return false;
    } 

    pub fn get_value(&mut self, name: &str) -> Option<Value> {
        for frame in self.frames.last_mut().unwrap().iter().rev() {
            match frame.get(name) {
                Some(v) => return Some(v.clone()),
                _ => ()
            }
        }
        return None;
    }

    pub fn new_frame(&mut self) {
        self.frames.last_mut().unwrap().push(HashMap::new());
    }

    pub fn drop_frame(&mut self) {
        self.frames.last_mut().unwrap().pop();
    }

    pub fn new_function_context(&mut self) {
        self.frames.push(Vec::new());
        self.new_frame();
    }

    pub fn drop_function_context(&mut self) {
        self.frames.pop();
    }
}

#[derive(Debug)]
pub struct State {
    pub stack: Stack, 
    pub heap: Heap
}

impl State {
    pub fn new() -> Self {
        State {
            stack: Stack::new(),
            heap: Heap::new()
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    VariableNotFound(String),
    FunctionNotFound(String),
    KeyNotFound,
    IndexOutOfBounds,
    UnhashableKey,
    // For binary operators. e.g., "Cannot apply '{op:?}' to types '{left}' and '{right}'"
    InvalidOperandTypesBin {
        op: ast::BinOp,
        left: &'static str,
        right: &'static str,
    },
    // For unary operators. e.g., "Cannot apply '{op:?}' to type '{operand}'"
    InvalidOperandTypesUn {
        op: ast::UnOp,
        operand: &'static str,
    },
    InvalidNamespace {},
    // For type mismatches, e.g., "Expected {expected}, got {got}"
    TypeError {
        expected: String,
        got: &'static str,
    },
    ConversionError {
        expected: String,
        got: String,
    },
    ArgumentError(String), // For missing/extra args
    ImmutabilityError(String), // For trying to assign to tuple/string
    InvalidAssignmentTarget,
    UnpackError(String), // For tuple/list unpacking
    MissingReturnValue,
    BlockError(String), // For "Expected Expression at end of block"
    InternalError(String), // For "Expected X Heap Object" - these are interpreter bugs
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpreterError::VariableNotFound(name) => write!(f, "Variable not found: '{}'", name),
            InterpreterError::FunctionNotFound(name) => write!(f, "Function not found: '{}'", name),
            InterpreterError::KeyNotFound => write!(f, "Key not found in dictionary"),
            InterpreterError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            InterpreterError::UnhashableKey => write!(f, "Unhashable key type"),
            InterpreterError::InvalidOperandTypesBin { op, left, right } => {
                write!(f, "Cannot apply operator '{:?}' to types '{}' and '{}'", op, left, right)
            },
            InterpreterError::InvalidOperandTypesUn { op, operand } => {
                write!(f, "Cannot apply operator '{:?}' to type '{}'", op, operand)
            },
            InterpreterError::InvalidNamespace {} => {
                write!(f, "invalid namespace")
            },
            InterpreterError::TypeError { expected, got } => {
                write!(f, "Type error: expected {}, got {}", expected, got)
            },
            InterpreterError::ConversionError { expected, got } => {
                write!(f, "Conversion error: expected {}, got {}", expected, got)
            },
            InterpreterError::ArgumentError(msg) => write!(f, "Argument error: {}", msg),
            InterpreterError::ImmutabilityError(msg) => write!(f, "Immutability error: {}", msg),
            InterpreterError::InvalidAssignmentTarget => write!(f, "Invalid assignment target"),
            InterpreterError::UnpackError(msg) => write!(f, "Unpack error: {}", msg),
            InterpreterError::MissingReturnValue => write!(f, "Function did not return a value"),
            InterpreterError::BlockError(msg) => write!(f, "Block error: {}", msg),
            InterpreterError::InternalError(msg) => write!(f, "Interpreter internal error: {}. This is likely a bug.", msg),
        }
    }
}


#[derive(Debug, PartialEq)]
pub struct InterpreterErrorMessage {
    pub error: InterpreterError,
    pub loc: Option<ast::Loc>
}

// we return Ok(Ok(Value)) if we just evaluate
// we short-circuit with Ok(Err(Value)) as this means we have a direct return value
pub fn eval_expression(state: &mut State, expression: &ast::LocExpr, program: &ast::Program) -> Result<Result<Value, Value>, InterpreterErrorMessage> {
    
    match expression.expr {
        ast::Expr::Variable(ref v) => {
            match state.stack.get_value(&v) {
                Some(value) => return Ok(Ok(value)),
                _ => ()
            }

            match program.functions.get(v) {
                Some(_) => return Ok(Ok(Value::FunctionPtr(v.clone()))),
                _ => ()
            }

            match v.as_str() {
                "Int" => return Ok(Ok(Value::NameSpacePtr(String::from("Int")))),
                "Bool" => return Ok(Ok(Value::NameSpacePtr(String::from("Bool")))),
                "String" => return Ok(Ok(Value::NameSpacePtr(String::from("String")))),
                "Tuple" => return Ok(Ok(Value::NameSpacePtr(String::from("Tuple")))),
                "List" => return Ok(Ok(Value::NameSpacePtr(String::from("List")))),
                "Dict" => return Ok(Ok(Value::NameSpacePtr(String::from("Dict")))),
                _ => ()
            }

            match v.as_str() {
                "int" => return Ok(Ok(Value::FunctionPtr(String::from("int")))),
                "bool" => return Ok(Ok(Value::FunctionPtr(String::from("bool")))),
                "str" => return Ok(Ok(Value::FunctionPtr(String::from("str")))),
                "tuple" => return Ok(Ok(Value::FunctionPtr(String::from("tuple")))),
                "list" => return Ok(Ok(Value::FunctionPtr(String::from("list")))),
                "dict" => return Ok(Ok(Value::FunctionPtr(String::from("dict")))),

                "len" => return Ok(Ok(Value::FunctionPtr(String::from("len")))),
                "print" => return Ok(Ok(Value::FunctionPtr(String::from("print")))),

                _ => ()
            }

            Err(InterpreterErrorMessage {
                error: InterpreterError::VariableNotFound(v.clone()),
                loc: Some(expression.loc.clone())
            })
        },
        ast::Expr::DotAccess(ref e, ref v) => {
            let value = match eval_expression(state, &e, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };

            let namespace = match value {
                Value::NameSpacePtr(s) => s,
                x => Value::get_type_namespace(&x, &e.loc)?
            };

            // todo: accesses
            Ok(Ok(Value::FunctionPtr(format!("{}.{}", namespace, v))))
        },
        ast::Expr::Int(ref i) => Ok(Ok(Value::Int(*i))),
        ast::Expr::Bool(ref b) => Ok(Ok(Value::Bool(*b))),
        ast::Expr::Str(ref s) => {
            let ptr = state.heap.intern_string(String::from(s));
            Ok(Ok(Value::String(ptr)))
        },
        ast::Expr::Tuple(ref values) => {
            let values: Result<Result<Vec<Value>, Value>, InterpreterErrorMessage>
                = values.into_iter().map(|arg| eval_expression(state, &arg, program)).collect();
            let values: Result<Vec<Value>, Value> = values?;
            match values {
                Ok(values) => Ok(Ok(Value::Tuple(values))),
                Err(value) => Ok(Err(value))
            }
        },
        ast::Expr::List(ref values) => {
            let values: Result<Result<Vec<Value>, Value>, InterpreterErrorMessage>
                = values.into_iter().map(|arg| eval_expression(state, &arg, program)).collect();
            let values: Result<Vec<Value>, Value> = values?;
            match values {
                Ok(values) => {
                    let ptr = state.heap.alloc(HeapObject::List(values));
                    Ok(Ok(Value::List(ptr)))
                },
                Err(value) => Ok(Err(value))
            }
        },
        ast::Expr::Dictionary(ref keys_values) => {
            let results: Result<Result<Vec<((Value, ast::Loc), (Value, ast::Loc))>, Value>, InterpreterErrorMessage> =
                keys_values.into_iter().map(|(key_expr, value_expr)| {
                    let key_result: Result<Value, Value> = eval_expression(state, key_expr, program)?;
                    let key_value = match key_result {
                        Ok(v) => v,
                        Err(v) => return Ok(Err(v)),
                    };
                    let value_result: Result<Value, Value> = eval_expression(state, value_expr, program)?;
                    let value_value = match value_result {
                        Ok(v) => v,
                        Err(v) => return Ok(Err(v)),
                    };
                    Ok(Ok(((key_value, key_expr.loc.clone()),
                           (value_value, value_expr.loc.clone()))))

                }).collect();
            let keys_values: Result<Vec<((Value, ast::Loc), (Value, ast::Loc))>, Value> = results?;

            match keys_values {
                Ok(keys_values) => {
                    for ((key, key_loc), (_, _)) in keys_values.iter() {
                        if !key.hashable() {
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::UnhashableKey,
                                loc: Some(key_loc.clone())
                            })
                        }
                    }

                    let mut map = HashMap::new();
                    for ((key, _), (value, _)) in keys_values {
                        map.insert(key, value);
                    }

                    let ptr = state.heap.alloc(HeapObject::Dictionary(map));
                    Ok(Ok(Value::Dictionary(ptr)))
                },
                Err(value) => Ok(Err(value))
            }

            
        },
        ast::Expr::BinOp { ref op, ref left, ref right } => {
            let left_value = match eval_expression(state, &left, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };
            let right_value = match eval_expression(state, &right, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };

            match (left_value.clone(), right_value.clone()) {
                (Value::Int(left_value), Value::Int(right_value)) => {
                    match op {
                        ast::BinOp::Eq => return Ok(Ok(Value::Bool(left_value == right_value))),
                        ast::BinOp::Neq => return Ok(Ok(Value::Bool(left_value != right_value))),
                        ast::BinOp::Leq => return Ok(Ok(Value::Bool(left_value <= right_value))),
                        ast::BinOp::Geq => return Ok(Ok(Value::Bool(left_value >= right_value))),
                        ast::BinOp::Lt => return Ok(Ok(Value::Bool(left_value < right_value))),
                        ast::BinOp::Gt => return Ok(Ok(Value::Bool(left_value > right_value))),
                        ast::BinOp::Add => return Ok(Ok(Value::Int(left_value + right_value))),
                        ast::BinOp::Sub => return Ok(Ok(Value::Int(left_value - right_value))),
                        ast::BinOp::Mul => return Ok(Ok(Value::Int(left_value * right_value))),
                        ast::BinOp::Div => return Ok(Ok(Value::Int(left_value / right_value))),
                        ast::BinOp::Mod => return Ok(Ok(Value::Int(left_value % right_value))),
                        ast::BinOp::ShiftLeft => return Ok(Ok(Value::Int(left_value << right_value))),
                        ast::BinOp::ShiftRightArith => return Ok(Ok(Value::Int(left_value >> right_value))),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { op: op.clone(), left: "int", right: "int" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                (Value::Bool(left_value), Value::Bool(right_value)) => {
                    match op {
                        ast::BinOp::Eq => return Ok(Ok(Value::Bool(left_value == right_value))),
                        ast::BinOp::Neq => return Ok(Ok(Value::Bool(left_value != right_value))),
                        ast::BinOp::And => return Ok(Ok(Value::Bool(left_value && right_value))),
                        ast::BinOp::Or => return Ok(Ok(Value::Bool(left_value || right_value))),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { op: op.clone(), left: "bool", right: "bool" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                (Value::List(left_ptr), Value::List(right_ptr)) => {
                    let left_value = match state.heap.get(left_ptr) {
                        Some(HeapObject::List(l)) => l,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                            loc: Some(expression.loc.clone())
                        })
                    };

                    let right_value = match state.heap.get(right_ptr) {
                        Some(HeapObject::List(l)) => l,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                            loc: Some(expression.loc.clone())
                        })
                    };

                    match op {
                        ast::BinOp::Add => {
                            let mut new_list = left_value.clone();
                            new_list.extend(right_value.clone());
                            let ptr = state.heap.alloc(HeapObject::List(new_list));
                            Ok(Ok(Value::List(ptr)))
                        },
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { op: op.clone(), left: "list", right: "list" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { 
                                op: op.clone(), 
                                left: left_value.get_type_name(), 
                                right: right_value.get_type_name() 
                            },
                            loc: Some(expression.loc.clone())
                        })
            }
        },
        ast::Expr::UnOp { ref op, ref expr } => {
            let value = match eval_expression(state, &expr, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };

            match value.clone() {
                Value::Int(value) => {
                    match op {
                        ast::UnOp::Neg => return Ok(Ok(Value::Int(-value))),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesUn { op: op.clone(), operand: "int" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                Value::Bool(value) => {
                    match op {
                        ast::UnOp::Not => return Ok(Ok(Value::Bool(!value))),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesUn { op: op.clone(), operand: "bool" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesUn { op: op.clone(), operand: value.get_type_name() },
                            loc: Some(expression.loc.clone())
                        })
            }
        },
        ast::Expr::FunctionCall { 
            ref function,
            ref positional_arguments,
            ref variadic_argument,
            ref keyword_arguments,
            ref keyword_variadic_argument 
        } => {
            if let ast::Expr::DotAccess(ref base_expr, ref method_name) = function.expr {
                let self_value = match eval_expression(state, base_expr, program)? {
                    Ok(value) => value,
                    Err(value) => return Ok(Err(value))
                };

                let (function_name, new_positional_args) = match self_value {
                    Value::NameSpacePtr(namespace) => {
                        // This is a STATIC call, e.g., String.len("abc")
                        let function_name = format!("{}.{}", namespace, method_name);
                        (function_name, positional_arguments.clone())
                    },
                    _ => {
                        // This is an INSTANCE call, e.g., s.len()
                        let namespace = Value::get_type_namespace(&self_value, &base_expr.loc)?;
                        let function_name = format!("{}.{}", namespace, method_name);

                        let self_arg = ast::CallArgument {
                            expr: base_expr.clone(),
                            loc: base_expr.loc.clone()
                        };
                        
                        let mut new_args = positional_arguments.clone();
                        new_args.insert(0, self_arg); 
                        (function_name, new_args)
                    }
                };

                return call_function(
                    state,
                    &function_name,
                    &expression.loc,
                    &new_positional_args, 
                    variadic_argument,
                    keyword_arguments,
                    keyword_variadic_argument,
                    program
                );

            } else {
                // This is a NORMAL function call, e.g., print(x) or a lambda call
                let func_value = match eval_expression(state, function, program)? {
                    Ok(value) => value,
                    Err(value) => return Ok(Err(value))
                };

                match func_value {
                    Value::FunctionPtr(ptr) => 
                        call_function(
                            state,
                            &ptr,
                            &expression.loc,
                            positional_arguments,
                            variadic_argument,
                            keyword_arguments,
                            keyword_variadic_argument,
                            program
                        ),
                    Value::Lambda(ptr) => {
                        let (arguments, expr) = match state.heap.get(ptr) {
                            Some(HeapObject::Lambda { arguments, expr }) => {
                                (arguments.clone(), expr.clone())
                            },
                            _ => return Err(InterpreterErrorMessage {
                                error: InterpreterError::InternalError("Expected Lambda Heap Object".to_string()),
                                loc: Some(function.loc.clone())
                            })
                        };

                        let argument_values: Result<Result<Vec<Value>, Value>, InterpreterErrorMessage>
                            = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expr, program)).collect();


                        let argument_values: Vec<Value> = match argument_values? {
                            Ok(values) => values,
                            Err(value) => return Ok(Err(value))
                        };

                        if argument_values.len() < arguments.len() {
                            let pos = argument_values.len();
                            let missing_arg = arguments.get(pos).unwrap();
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::ArgumentError(format!("Missing argument: '{}'", missing_arg.name)),
                                loc: Some(missing_arg.loc.clone())
                            })
                        }

                        if argument_values.len() > arguments.len() {
                            let pos = arguments.len();
                            
                            let extra_arg_expression = match pos < positional_arguments.len() {
                                true => positional_arguments.get(pos).unwrap(),
                                false => &variadic_argument.clone().unwrap()
                            };

                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::ArgumentError("Unexpected positional argument".to_string()),
                                loc: Some(extra_arg_expression.loc.clone())
                            })
                        }

                        let new_values: HashMap<String, Value> = arguments.iter().zip(argument_values.iter()).map(|(arg, value)| (arg.name.clone(), value.clone())).collect();

                        state.stack.new_function_context();

                        let _ = new_values.into_iter().for_each(|(n,v)| {state.stack.update_variable(&n, v);});
                        
                        let value: Result<Result<Value, Value>, InterpreterErrorMessage> = match eval_expression(state, &expr, program) {
                            Ok(Ok(value)) | Ok(Err(value)) => Ok(Ok(value)),
                            Err(value) => return Err(value)
                        };

                        state.stack.drop_function_context();

                        return value;
                    },
                    _ => Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError { 
                                expected: "callable (function or lambda)".to_string(), 
                                got: func_value.get_type_name()
                            },
                            loc: Some(function.loc.clone())
                        })
                }
            }
        },
        ast::Expr::Indexing { ref indexed, ref indexer } => {
            let original_indexed_value = match eval_expression(state, &indexed, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };
            let original_indexer_value = match eval_expression(state, &indexer, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };

            if let Value::Dictionary(ptr) = original_indexed_value {
                match state.heap.get(ptr) {
                    Some(HeapObject::Dictionary(dict)) => {
                        if !original_indexer_value.hashable() {
                             return Err(InterpreterErrorMessage {
                                error: InterpreterError::UnhashableKey,
                                loc: Some(indexer.loc.clone())
                            })
                        }
                        match dict.get(&original_indexer_value) {
                            Some(value) => return Ok(Ok(value.clone())),
                            _ => return Err(InterpreterErrorMessage {
                                error: InterpreterError::KeyNotFound,
                                loc: Some(indexer.loc.clone())
                            })
                        }
                    },
                    _ => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                            loc: Some(indexed.loc.clone())
                        })
                    }
                }
            }

            let mut indexer_value = match original_indexer_value.clone() {
                Value::Int(i) => i,
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { 
                        expected: "int".to_string(), 
                        got: original_indexer_value.get_type_name() 
                    },
                    loc: Some(indexer.loc.clone())
                })
            };

            let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;

            if indexer_value < 0 {
                indexer_value = (indexed_length as i64) + indexer_value;
            }
            

            if indexer_value < 0 || indexer_value >= (indexed_length as i64) {
                return Err(InterpreterErrorMessage {
                    error: InterpreterError::IndexOutOfBounds,
                    loc: Some(indexer.loc.clone())
                });
            }

            let index = indexer_value as usize;

            match original_indexed_value {
                Value::String(ptr) => {
                    match state.heap.get(ptr) {
                        Some(HeapObject::Str(str)) => {
                            let char_val = str.chars().nth(index).unwrap(); 
                            let char_str = char_val.to_string();
                            let new_ptr = state.heap.alloc(HeapObject::Str(char_str));
                            Ok(Ok(Value::String(new_ptr)))
                        },
                        _ => unreachable!() 
                    }
                },
                Value::Tuple(values) => {
                    Ok(Ok(values[index].clone()))
                },
                Value::List(ptr) => {
                    match state.heap.get(ptr) {
                    Some(HeapObject::List(l)) => {
                        Ok(Ok(l[index].clone()))
                    },
                    _ => unreachable!()
                    }
                },
                _ => unreachable!()
            }
        },
        ast::Expr::FunctionPtr(ref s) => {Ok(Ok(Value::FunctionPtr(s.clone())))},
        ast::Expr::Lambda { ref arguments, ref expr } => {
            let ptr = state.heap.alloc(HeapObject::Lambda {arguments: arguments.clone(), expr: expr.clone()});
            Ok(Ok(Value::Lambda(ptr)))
        },
        ast::Expr::Block { ref statements } => {
            match statements.as_slice() {
                [rest @ .., last] => {
                    
                    state.stack.new_frame();

                    for stmt in rest.iter() {
                        let ret = run_statement(state, stmt, program);
                        match ret {
                            Err(e) => {
                                return Err(e)
                            },
                            Ok(StatementReturn::Return(v)) => {
                                state.stack.drop_frame();
                                return Ok(Err(v)) // we *return* a value
                            },
                            Ok(_) => {}
                        };
                    }

                    let value = match &last.stmt {
                        ast::Stmt::Expression { expr: expression } => {
                            eval_expression(state, &expression, program)
                        },
                        _ => Err(InterpreterErrorMessage {
                                error: InterpreterError::BlockError("Block used as expression must end with an expression".to_string()),
                                loc: Some(last.loc.clone())
                            })
                    };

                    state.stack.drop_frame();

                    value
                },
                [] => Err(InterpreterErrorMessage {
                                error: InterpreterError::BlockError("Block used as expression must end with an expression".to_string()),
                                loc: Some(expression.loc.clone())
                            })
            }
        }
    }
}


fn get_indexed_length(state: &mut State, original_indexed_value: &Value, indexed: &ast::LocExpr) -> Result<usize, InterpreterErrorMessage> {
    match &original_indexed_value {
        Value::String(ptr) => {
            match state.heap.get(*ptr) {
                Some(HeapObject::Str(str)) => Ok(str.chars().count()),
                _ => {
                    return Err(InterpreterErrorMessage {
                        error: InterpreterError::InternalError("Expected String Heap Object".to_string()),
                        loc: Some(indexed.loc.clone())
                    })
                }
            }
        },
        Value::Tuple(values) => Ok(values.len()),
        Value::List(ptr) => {
            match state.heap.get(*ptr) {
                Some(HeapObject::List(l)) => Ok(l.len()),
                _ => {
                    return Err(InterpreterErrorMessage {
                        error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                        loc: Some(indexed.loc.clone())
                    })
                }
            }
        },
        _ => return Err(InterpreterErrorMessage {
            error: InterpreterError::TypeError { 
                expected: "indexable (string, tuple, list, dict)".to_string(), 
                got: original_indexed_value.get_type_name()
            },
            loc: Some(indexed.loc.clone())
        })
    }
}


fn preprocess_args(
    state: &mut State,
    contract: &ast::FunctionPrototype,
    loc: &ast::Loc,
    positional_arguments: &Vec<ast::CallArgument>,
    variadic_argument: &Option<ast::CallArgument>,
    keyword_arguments: &Vec<ast::CallKeywordArgument>,
    keyword_variadic_argument: &Option<ast::CallArgument>,
    program: &ast::Program
) -> Result<Result<HashMap<String, Value>, Value>, InterpreterErrorMessage> {
    
    let argument_values: Result<Result<Vec<Value>, Value>, InterpreterErrorMessage>
        = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expr, program)).collect();

    let mut argument_values: Vec<Value> = match argument_values? {
        Ok(values) => values,
        Err(value) => return Ok(Err(value))
    };

    match &variadic_argument {
        Some(arg) => {
            let value = match eval_expression(state, &arg.expr, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };
            let extra_args: Vec<Value> = match value.clone() {
                Value::Tuple(elements) => elements,
                Value::List(ptr) => {
                    match state.heap.get(ptr) {
                        Some(HeapObject::List(list)) => list.clone(),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                            loc: Some(arg.loc.clone())
                        })
                    }
                },
                x => return Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError {
                        expected: "tuple or list".to_string(),
                        got: value.get_type_name()
                    },
                    loc: Some(arg.loc.clone())
                })
            };
            argument_values.extend(extra_args);
        },
        _ => ()
    }

    let keyword_values: Result<Result<HashMap<String, (Option<&ast::CallKeywordArgument>, Value)>, Value>, InterpreterErrorMessage> = keyword_arguments.iter()
        .map(|arg| {
            match eval_expression(state, &arg.expr, program) {
                Ok(Ok(value)) => Ok(Ok((arg.name.clone(), (Some(arg), value)))),
                Ok(Err(value)) => Ok(Err(value)),
                Err(e) => Err(e)
            }
        })
        .collect();

    let mut keyword_values: HashMap<String, (Option<&ast::CallKeywordArgument>, Value)> = match keyword_values? {
        Ok(values) => values,
        Err(value) => return Ok(Err(value))
    };

    match keyword_variadic_argument {
        Some(arg) => {
            let value = match eval_expression(state, &arg.expr, program)? {
                Ok(value) => value,
                Err(value) => return Ok(Err(value))
            };
            match value.clone() {
                Value::Dictionary(ptr) => {
                    let index_ref = match state.heap.get(ptr) {
                        Some(HeapObject::Dictionary(d)) => d,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                            loc: Some(arg.loc.clone())
                        })
                    };

                    for (key, value) in index_ref.iter() {
                        match key.clone() {
                            Value::String(ptr) => {
                                let s = match state.heap.get(ptr) {
                                    Some(HeapObject::Str(s)) => s,
                                    _ => return Err(InterpreterErrorMessage {
                                        error: InterpreterError::InternalError("Expected String Heap Object".to_string()),
                                        loc: Some(arg.loc.clone())
                                    })
                                };

                                if !keyword_values.contains_key(&s.clone()) {
                                    keyword_values.insert(s.clone(), (None, value.clone()));
                                }
                            },
                            x => return Err(InterpreterErrorMessage {
                                error: InterpreterError::TypeError {
                                    expected: "string key".to_string(),
                                    got: key.get_type_name()
                                },
                                loc: Some(arg.loc.clone())
                            })                               
                        }
                    }
                },
                x => return Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError {
                        expected: "dict".to_string(),
                        got: value.get_type_name()
                    },
                    loc: Some(arg.loc.clone())
                })
            };
        },
        _ => ()
    }

    if argument_values.len() < contract.positional_arguments.len() {
        let pos = argument_values.len();
        let missing_arg = contract.positional_arguments.get(pos).unwrap();
        return Err(InterpreterErrorMessage {
            error: InterpreterError::ArgumentError(format!("Missing argument: '{}'", missing_arg.name)),
            loc: Some(missing_arg.loc.clone())
        })
    }

    if contract.variadic_argument.is_none() && argument_values.len() > contract.positional_arguments.len() {
        let pos = contract.positional_arguments.len();
        
        let extra_arg_expression = match pos < positional_arguments.len() {
            true => positional_arguments.get(pos).unwrap(),
            false => &variadic_argument.clone().unwrap()
        };

        return Err(InterpreterErrorMessage {
            error: InterpreterError::ArgumentError("Unexpected positional argument".to_string()),
            loc: Some(extra_arg_expression.loc.clone())
        })
    }

    let mut new_values: HashMap<String, Value> = contract.positional_arguments.iter().zip(argument_values.iter()).map(|(arg, value)| (arg.name.clone(), value.clone())).collect();

    if argument_values.len() > contract.positional_arguments.len() {
        // from previous logic the only way this happens if we have a variadic accepting argument in the function
        let extra_args = &argument_values[contract.positional_arguments.len()..];
        let ptr = state.heap.alloc(HeapObject::List(extra_args.to_vec()));

        new_values.insert(contract.variadic_argument.clone().unwrap().name, Value::List(ptr));
    }

    let mut keyword_variadic_arguments: HashMap<Value, Value> = HashMap::new();

    for keyword_arg in contract.keyword_arguments.clone() {
        let value = match eval_expression(state, &keyword_arg.expr, program)? {
            Ok(value) => value,
            Err(value) => return Ok(Err(value))
        }; 
        new_values.insert(keyword_arg.name, value);
    }

    for (key, (arg, value)) in keyword_values {
        match contract.keyword_arguments.iter().filter(|y| y.name == key).peekable().peek() {
            Some(_) => {
                new_values.insert(key.clone(), value);
            }
            _ => {
                match &contract.keyword_variadic_argument {
                    Some(_) => {
                        let ptr = state.heap.alloc(HeapObject::Str(String::from(key)));
                        keyword_variadic_arguments.insert(Value::String(ptr), value);
                    }
                    _ => match arg {
                        Some(arg) => {
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::ArgumentError(format!("Unexpected keyword argument: '{}'", arg.name)),
                                loc: Some(arg.loc.clone())
                            })
                        },
                        _ => {
                            // If we do not have an arg passed this means that it originally came from the keyword_variadic argument
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::ArgumentError(format!("Unexpected keyword argument: '{}'", key)),
                                loc: Some(keyword_variadic_argument.as_ref().unwrap().loc.clone())
                            })
                        },
                    }
                }
            }
        }
    }

    match &contract.keyword_variadic_argument {
        Some(arg) => {
            let ptr = state.heap.alloc(HeapObject::Dictionary(keyword_variadic_arguments));
            new_values.insert(arg.name.clone(), Value::Dictionary(ptr) );
        },
        _ => assert!(keyword_variadic_arguments.is_empty())
    };

    Ok(Ok(new_values))
}

fn call_function(
    state: &mut State,
    function_name: &str,
    loc: &ast::Loc,
    positional_arguments: &Vec<ast::CallArgument>,
    variadic_argument: &Option<ast::CallArgument>,
    keyword_arguments: &Vec<ast::CallKeywordArgument>,
    keyword_variadic_argument: &Option<ast::CallArgument>,
    program: &ast::Program
) -> Result<Result<Value, Value>, InterpreterErrorMessage> {
    let function = match program.functions.get(function_name) {
        Some(f) => f,
        _ => return call_builtin(state, function_name, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)
    };

    let new_values = preprocess_args(state, &function.contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;

    let new_values = match new_values {
        Ok(v) => v,
        Err(v) => return Ok(Err(v))
    };

    state.stack.new_function_context();

    new_values.into_iter().for_each(|(n,v)| {state.stack.update_variable(&n, v);});

    let value = run_statement(state, &function.body, program);

    state.stack.drop_function_context();

    match value {
        Ok(StatementReturn::Return(v)) => {
            Ok(Ok(v))
        },
        Ok(_) => {
            return Err(InterpreterErrorMessage {
                error: InterpreterError::MissingReturnValue,
                loc: Some(loc.clone())
            })
        },
        Err(e) => return Err(e)
    }
}

fn get_len(value: &Value, heap: &Heap, arg_loc: &ast::Loc) -> Result<i64, InterpreterErrorMessage> {
    match value {
        Value::String(ptr) => {
            match heap.get(*ptr) {
                Some(HeapObject::Str(s)) => Ok(s.chars().count() as i64),
                _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("String expected in heap")), loc: Some(arg_loc.clone())})
            }
        },
        Value::Tuple(v) => Ok(v.len() as i64),
        Value::List(ptr) => {
            match heap.get(*ptr) {
                Some(HeapObject::List(l)) => Ok(l.len() as i64),
                 _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected in heap")), loc: Some(arg_loc.clone())})
            }
        },
        Value::Dictionary(ptr) => {
             match heap.get(*ptr) {
                Some(HeapObject::Dictionary(d)) => Ok(d.len() as i64),
                 _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("Dictionary expected in heap")), loc: Some(arg_loc.clone())})
            }
        },
        _ => Err(InterpreterErrorMessage {
            error: InterpreterError::TypeError {
                expected: "string, tuple, list, or dict".to_string(),
                got: value.get_type_name()
            },
            loc: Some(arg_loc.clone())
        })
    }
}

fn deep_clone_value(state: &mut State, value: &Value) -> Result<Value, InterpreterErrorMessage> {
    match value {
        Value::Void |
        Value::Int(_) |
        Value::Bool(_) |
        Value::FunctionPtr(_) |
        Value::NameSpacePtr(_) => Ok(value.clone()),

        Value::String(ptr) => {
            let s = match state.heap.get(*ptr) {
                Some(HeapObject::Str(s)) => s.clone(),
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::InternalError("Expected String Heap Object".to_string()),
                    loc: None // We don't have location context here
                })
            };
            let new_ptr = state.heap.intern_string(s);
            Ok(Value::String(new_ptr))
        },

        Value::Tuple(values) => {
            let new_values = values.iter()
                .map(|v| deep_clone_value(state, v))
                .collect::<Result<Vec<Value>, _>>()?;
            Ok(Value::Tuple(new_values))
        },

        Value::List(ptr) => {
            let list = match state.heap.get(*ptr) {
                Some(HeapObject::List(l)) => l.clone(), // Clone the Vec (shallow)
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                    loc: None
                })
            };
            let new_list = list.iter()
                .map(|v| deep_clone_value(state, v))
                .collect::<Result<Vec<Value>, _>>()?;
            
            let new_ptr = state.heap.alloc(HeapObject::List(new_list));
            Ok(Value::List(new_ptr))
        },

        Value::Dictionary(ptr) => {
            let dict = match state.heap.get(*ptr) {
                Some(HeapObject::Dictionary(d)) => d.clone(), // Clone the HashMap (shallow)
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                    loc: None
                })
            };
            let mut new_dict = HashMap::new();
            for (k, v) in dict.iter() {
                let new_k = deep_clone_value(state, k)?;
                let new_v = deep_clone_value(state, v)?;
                new_dict.insert(new_k, new_v);
            }

            let new_ptr = state.heap.alloc(HeapObject::Dictionary(new_dict));
            Ok(Value::Dictionary(new_ptr))
        },

        Value::Lambda(ptr) => {
            let lambda_obj = match state.heap.get(*ptr) {
                Some(HeapObject::Lambda {..}) => state.heap.get(*ptr).unwrap().clone(),
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::InternalError("Expected Lambda Heap Object".to_string()),
                    loc: None
                })
            };
            let new_ptr = state.heap.alloc(lambda_obj);
            Ok(Value::Lambda(new_ptr))
        }
    }
}

fn call_builtin(
    state: &mut State,
    function_name: &str,
    loc: &ast::Loc,
    positional_arguments: &Vec<ast::CallArgument>,
    variadic_argument: &Option<ast::CallArgument>,
    keyword_arguments: &Vec<ast::CallKeywordArgument>,
    keyword_variadic_argument: &Option<ast::CallArgument>,
    program: &ast::Program
) -> Result<Result<Value, Value>, InterpreterErrorMessage> {

    match function_name {
        "int" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args_map.get("args").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => l,
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };
            
            let result_value = if values.is_empty() {
                // Case 1: int() -> 0
                Value::Int(0)
            } else {
                // Case 2: int(x)
                let value_to_convert = &values[0];
                match value_to_convert {
                    Value::Int(i) => Value::Int(*i),
                    Value::Bool(b) => Value::Int(if *b { 1 } else { 0 }),
                    Value::String(ptr) => {
                        // Retrieve string content from the heap
                        let s = match state.heap.get(*ptr) {
                            Some(HeapObject::Str(s)) => s,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("String expected in heap for Value::String")), loc: Some(loc.clone())})
                        };
                        
                        // Attempt to parse the string
                        match s.parse::<i64>() {
                            Ok(i) => Value::Int(i),
                            Err(_) => {
                                return Err(InterpreterErrorMessage {
                                    error: InterpreterError::ConversionError { 
                                        expected: "int".to_string(), 
                                        got: s.clone()
                                    },
                                    loc: Some(loc.clone())
                                })
                            }
                        }
                    },
                    _ => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::ConversionError { 
                                expected: "int".to_string(), 
                                got: format!("{:?}", value_to_convert)
                            },
                            loc: Some(loc.clone())
                        })
                    }
                }
            };
            
            return Ok(Ok(result_value));
        },
        "bool" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args_map.get("args").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => l,
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };

            let result_value = if values.is_empty() {
                // Case 1: bool() -> False
                Value::Bool(false)
            } else {
                // Case 2: bool(x)
                let value_to_convert = &values[0];

                match value_to_convert {
                    Value::Bool(b) => Value::Bool(*b),
                    Value::Int(i) => Value::Bool(*i != 0),
                    Value::Void => Value::Bool(false),
                    Value::String(ptr) => {
                        // Retrieve string content from the heap
                        let s = match state.heap.get(*ptr) {
                            Some(HeapObject::Str(s)) => s,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("String expected in heap for Value::String")), loc: Some(loc.clone())})
                        };
                        Value::Bool(!s.is_empty())
                    },
                    Value::Tuple(v) => Value::Bool(!v.is_empty()),
                    Value::List(ptr) => {
                        // Retrieve list content from the heap
                        let l = match state.heap.get(*ptr) {
                            Some(HeapObject::List(l)) => l,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected in heap for Value::List")), loc: Some(loc.clone())})
                        };
                        Value::Bool(!l.is_empty())
                    },
                    Value::Dictionary(ptr) => {
                        // Retrieve dictionary content from the heap (assuming `HeapObject::Dict(d)` yields a reference with `is_empty()`)
                        let d = match state.heap.get(*ptr) {
                            Some(HeapObject::Dictionary(d)) => d,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("Dictionary expected in heap for Value::Dictionary")), loc: Some(loc.clone())})
                        };
                        Value::Bool(!d.is_empty())
                    },
                    // Functions and Lambdas are generally truthy
                    Value::Lambda(_) | Value::FunctionPtr(_) | Value::NameSpacePtr(_) => Value::Bool(true),
                }
            };
            
            return Ok(Ok(result_value));
        },
        "str" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args_map.get("args").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => l,
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };

            let result_string = if values.is_empty() {
                // Case 1: str() -> ""
                String::from("")
            } else {
                // Case 2: str(x)
                let value_to_convert = &values[0];
                DisplayValue { heap: &state.heap, value: value_to_convert, is_contained: false }.to_string()
            };

            let ptr = state.heap.intern_string(result_string);
            
            return Ok(Ok(Value::String(ptr)));
        },
        "tuple" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args_map.get("args").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => l,
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };

            let iterable_elements: Vec<Value>;

            if values.is_empty() {
                // Case 1: tuple() -> ()
                iterable_elements = vec![];
            } else if values.len() == 1 {
                // Case 2: tuple(x) - Convert iterable x
                let value_to_convert = &values[0];

                match value_to_convert {
                    Value::Tuple(v) => {
                        iterable_elements = v.clone();
                    },
                    Value::List(ptr) => {
                        let l = match state.heap.get(*ptr) {
                            Some(HeapObject::List(l)) => l,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected in heap for Value::List")), loc: Some(loc.clone())})
                        };
                        iterable_elements = l.clone();
                    },
                    Value::String(ptr) => {
                        let s = match state.heap.get(*ptr) {
                            Some(HeapObject::Str(s)) => s.clone(),
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("String expected in heap for Value::String")), loc: Some(loc.clone())})
                        };
                        iterable_elements = s.chars().map(|c| {
                            let s_ptr = state.heap.intern_string(c.to_string());
                            Value::String(s_ptr)
                        }).collect();
                    },
                    Value::Dictionary(ptr) => {
                        let d = match state.heap.get(*ptr) {
                            Some(HeapObject::Dictionary(d)) => d,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("Dictionary expected in heap for Value::Dictionary")), loc: Some(loc.clone())})
                        };
                        iterable_elements = d.keys().cloned().collect();
                    },
                    _ => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::ConversionError { 
                                expected: "iterable".to_string(), 
                                got: format!("{:?}", value_to_convert)
                            },
                            loc: Some(loc.clone())
                        })
                    }
                }
            } else {
                iterable_elements = values.clone();
            }

            return Ok(Ok(Value::Tuple(iterable_elements)));
        },
        "list" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args_map.get("args").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => l,
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };
            
            let iterable_elements: Vec<Value>;

            if values.is_empty() {
                // Case 1: list() -> []
                iterable_elements = vec![];
            } else if values.len() == 1 {
                // Case 2: list(x) - Convert iterable x
                let value_to_convert = &values[0];

                match value_to_convert {
                    Value::Tuple(v) => {
                        iterable_elements = v.clone();
                    },
                    Value::List(ptr) => {
                        let l = match state.heap.get(*ptr) {
                            Some(HeapObject::List(l)) => l,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected in heap for Value::List")), loc: Some(loc.clone())})
                        };
                        iterable_elements = l.clone();
                    },
                    Value::String(ptr) => {
                        let s = match state.heap.get(*ptr) {
                            Some(HeapObject::Str(s)) => s.clone(),
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("String expected in heap for Value::String")), loc: Some(loc.clone())})
                        };
                        iterable_elements = s.chars().map(|c| {
                            let s_ptr = state.heap.intern_string(c.to_string());
                            Value::String(s_ptr)
                        }).collect();
                    },
                    Value::Dictionary(ptr) => {
                        let d = match state.heap.get(*ptr) {
                            Some(HeapObject::Dictionary(d)) => d,
                            _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("Dictionary expected in heap for Value::Dictionary")), loc: Some(loc.clone())})
                        };
                        iterable_elements = d.keys().cloned().collect();
                    },
                    _ => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::ConversionError { 
                                expected: "iterable".to_string(), 
                                got: format!("{:?}", value_to_convert)
                            },
                            loc: Some(loc.clone())
                        })
                    }
                }
            } else {
                iterable_elements = values.clone();
            }
            
            let ptr = state.heap.alloc(HeapObject::List(iterable_elements));
            
            return Ok(Ok(Value::List(ptr)));
        },
        "dict" => todo!(),

        "print" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("l"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args = match args {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args.get("l").unwrap() {
                Value::List(ptr) => match state.heap.get(*ptr) {
                    Some(HeapObject::List(l)) => {
                        l
                    },
                    _ => unreachable!()
                },
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected")), loc: Some(loc.clone())})
            };

            println!("{}", values.iter().map(|x| format!("{}", DisplayValue {heap: &state.heap, value: x, is_contained: false})).collect::<Vec<String>>().join(" "));
            return Ok(Ok(Value::Void))
        },
        "len" | "String.len" | "Tuple.len" | "List.len" | "Dict.len" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("obj"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let value = args_map.get("obj").unwrap();
            let arg_loc = &positional_arguments.get(0).unwrap().loc;

            return match get_len(value, &state.heap, arg_loc) {
                Ok(len) => Ok(Ok(Value::Int(len))),
                Err(e) => Err(e)
            };
        },

        "clone" | "Int.clone" | "Bool.clone" | "String.clone" | "Tuple.clone" | "List.clone" | "Dict.clone" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("obj"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let value = args_map.get("obj").unwrap();
            
            return match deep_clone_value(state, value) {
                Ok(cloned_value) => Ok(Ok(cloned_value)),
                Err(mut e) => {
                    if e.loc.is_none() {
                        e.loc = Some(positional_arguments.get(0).unwrap().loc.clone());
                    }
                    Err(e)
                }
            }
        }

        "List.push" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![
                    ast::Argument {name: String::from("l"), arg_type: None, loc: 0..0},
                    ast::Argument {name: String::from("v"), arg_type: None, loc: 0..0}
                ],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let l_val = args_map.get("l").unwrap();
            let v_val = args_map.get("v").unwrap();
            let l_loc = &positional_arguments.get(0).unwrap().loc;

            return match l_val {
                Value::List(ptr) => {
                    match state.heap.get_mut(*ptr) {
                        Some(HeapObject::List(list_vec)) => {
                            list_vec.push(v_val.clone());
                            Ok(Ok(Value::Void))
                        },
                        _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError("Expected List Heap Object".to_string()), loc: Some(l_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "list".to_string(), got: l_val.get_type_name() },
                    loc: Some(l_loc.clone())
                })
            }
        },
        "List.pop" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("l"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let l_val = args_map.get("l").unwrap();
            let l_loc = &positional_arguments.get(0).unwrap().loc;

            return match l_val {
                Value::List(ptr) => {
                    match state.heap.get_mut(*ptr) {
                        Some(HeapObject::List(list_vec)) => {
                            match list_vec.pop() {
                                Some(value) => Ok(Ok(value)),
                                _ => Err(InterpreterErrorMessage { error: InterpreterError::IndexOutOfBounds, loc: Some(loc.clone()) })
                            }
                        },
                        _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError("Expected List Heap Object".to_string()), loc: Some(l_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "list".to_string(), got: l_val.get_type_name() },
                    loc: Some(l_loc.clone())
                })
            }
        },

        "Dict.keys" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("d"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let d_val = args_map.get("d").unwrap();
            let d_loc = &positional_arguments.get(0).unwrap().loc;

            return match d_val {
                Value::Dictionary(ptr) => {
                    match state.heap.get(*ptr) {
                        Some(HeapObject::Dictionary(hash_map)) => {
                            let keys_vec = hash_map.keys().cloned().collect::<Vec<Value>>();
                            let new_list_ptr = state.heap.alloc(HeapObject::List(keys_vec));
                            Ok(Ok(Value::List(new_list_ptr)))
                        },
                        _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()), loc: Some(d_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "dict".to_string(), got: d_val.get_type_name() },
                    loc: Some(d_loc.clone())
                })
            }
        },
        "Dict.values" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("d"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let d_val = args_map.get("d").unwrap();
            let d_loc = &positional_arguments.get(0).unwrap().loc;

            return match d_val {
                Value::Dictionary(ptr) => {
                    match state.heap.get(*ptr) {
                        Some(HeapObject::Dictionary(hash_map)) => {
                            let values_vec = hash_map.values().cloned().collect::<Vec<Value>>();
                            let new_list_ptr = state.heap.alloc(HeapObject::List(values_vec));
                            Ok(Ok(Value::List(new_list_ptr)))
                        },
                        _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()), loc: Some(d_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "dict".to_string(), got: d_val.get_type_name() },
                    loc: Some(d_loc.clone())
                })
            }
        },
        "Dict.items" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("d"), arg_type: None, loc: 0..0}],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args_map = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args_map = match args_map {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let d_val = args_map.get("d").unwrap();
            let d_loc = &positional_arguments.get(0).unwrap().loc;

            return match d_val {
                Value::Dictionary(ptr) => {
                    match state.heap.get(*ptr) {
                        Some(HeapObject::Dictionary(hash_map)) => {
                            let items_vec = hash_map.iter()
                                .map(|(k, v)| Value::Tuple(vec![k.clone(), v.clone()]))
                                .collect::<Vec<Value>>();
                            let new_list_ptr = state.heap.alloc(HeapObject::List(items_vec));
                            Ok(Ok(Value::List(new_list_ptr)))
                        },
                        _ => Err(InterpreterErrorMessage {error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()), loc: Some(d_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "dict".to_string(), got: d_val.get_type_name() },
                    loc: Some(d_loc.clone())
                })
            }
        },

        _ => ()
    }



    return Err(InterpreterErrorMessage {
                error: InterpreterError::FunctionNotFound(function_name.to_string()),
                loc: Some(loc.clone())
            })
}



enum StatementReturn {
    Return(Value),
    Eval(Value),
    None
}

impl StatementReturn {

    pub fn unwrap(self: Self) -> Value {
        match self {
            Self::Return(v) => v,
            Self::Eval(v) => v,
            _ => panic!("unwrap failed")
        }
    }
}

pub fn run_statement(state: &mut State, stmt: &ast::LocStmt, program: &ast::Program) -> Result<StatementReturn, InterpreterErrorMessage> {
    match &stmt.stmt {
        ast::Stmt::Assignment { target, expr: expression } => {
            match &target.expr {
                ast::Expr::Variable(v) => {
                    let value = match eval_expression(state, expression, program)? {
                        Ok(value) => value,
                        Err(value) => return Ok(StatementReturn::Return(value))
                    };
                    state.stack.update_variable(&v,value);
                    return Ok(StatementReturn::None)
                },
                ast::Expr::Indexing { indexed, indexer } => {
                    let original_indexed_value = match eval_expression(state, &indexed, program)? {
                        Ok(value) => value,
                        Err(value) => return Ok(StatementReturn::Return(value))
                    };
                    let original_indexer_value = match eval_expression(state, &indexer, program)? {
                        Ok(value) => value,
                        Err(value) => return Ok(StatementReturn::Return(value))
                    };

                    let value = match eval_expression(state, expression, program)? {
                        Ok(value) => value,
                        Err(value) => return Ok(StatementReturn::Return(value))
                    };

                    if let Value::Dictionary(ptr) = original_indexed_value {
                        if !original_indexer_value.hashable() {
                             return Err(InterpreterErrorMessage {
                                error: InterpreterError::UnhashableKey,
                                loc: Some(indexer.loc.clone())
                            })
                        }
                        match state.heap.get_mut(ptr) {
                            Some(HeapObject::Dictionary(dict)) => {
                                dict.insert(original_indexer_value, value);
                            },
                            _ => {
                                return Err(InterpreterErrorMessage {
                                    error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                                    loc: Some(indexed.loc.clone())
                                })
                            }
                        }

                        return Ok(StatementReturn::None);
                    }

                    let mut indexer_value = match original_indexer_value.clone() {
                        Value::Int(i) => i,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError { 
                                expected: "int".to_string(), 
                                got: original_indexer_value.get_type_name() 
                            },
                            loc: Some(indexer.loc.clone())
                        })
                    };

                    let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;

                    if indexer_value < 0 {
                        indexer_value = (indexed_length as i64) + indexer_value;
                    }
                    

                    if indexer_value < 0 || indexer_value >= (indexed_length as i64) {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::IndexOutOfBounds,
                            loc: Some(indexer.loc.clone())
                        });
                    }

                    let index = indexer_value as usize;

                    match original_indexed_value {
                        Value::String(_) => {
                             return Err(InterpreterErrorMessage {
                                error: InterpreterError::ImmutabilityError("Strings are immutable".to_string()),
                                loc: Some(target.loc.clone())
                            });
                        },
                        Value::Tuple(_) => {
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::ImmutabilityError("Tuples are immutable".to_string()),
                                loc: Some(target.loc.clone())
                            });
                        },
                        Value::List(ptr) => {
                            match state.heap.get_mut(ptr) {
                                Some(HeapObject::List(l)) => {
                                    l[index] = value;
                                },
                                _ => unreachable!()
                            }
                        },
                        _ => unreachable!()
                    }
                },
                ast::Expr::Tuple(elements) | ast::Expr::List(elements) => {
                    let value = match eval_expression(state, expression, program)? {
                        Ok(value) => value,
                        Err(value) => return Ok(StatementReturn::Return(value))
                    };
                    let assignment_list: Vec<(String, Value)> = unpack_elements(state, elements, value, &expression.loc)?;
                    assignment_list.into_iter().for_each(|(var, value)| {
                        state.stack.update_variable(&var, value);
                    })
                },
                _ => return Err(InterpreterErrorMessage {
                        error: InterpreterError::InvalidAssignmentTarget,
                        loc: Some(target.loc.clone())
                    })
            }
        },
        ast::Stmt::FunctionCall { expr: expression } => {eval_expression(state, expression, program)?;},
        ast::Stmt::Return { expr: expression } => {
            let value = match eval_expression(state, expression, program)? {
                Ok(value) => value,
                Err(value) => return Ok(StatementReturn::Return(value))
            };
            return Ok(StatementReturn::Return(value));
        },
        ast::Stmt::IfElse { cond: condition, if_body, else_body } => {
            let eval_condition = match eval_expression(state, &condition, program)? {
                Ok(value) => value,
                Err(value) => return Ok(StatementReturn::Return(value))
            };

            match eval_condition {
                Value::Bool(b) => {
                    match b {
                        true => return run_statement(state, &if_body, program),
                        false => return run_statement(state, &else_body, program),
                    }
                }
                x => {
                    return Err(InterpreterErrorMessage {
                        error: InterpreterError::TypeError { 
                            expected: "bool".to_string(), 
                            got: x.get_type_name() 
                        },
                        loc: Some(condition.loc.clone())
                    });
                }
            };
        },
        ast::Stmt::While { cond: condition, body } => {
            let eval_condition = match eval_expression(state, condition, program)? {
                Ok(value) => value,
                Err(value) => return Ok(StatementReturn::Return(value))
            };

            let mut cond = match eval_condition {
                Value::Bool(b) => b,
                x => {
                    return Err(InterpreterErrorMessage {
                        error: InterpreterError::TypeError { 
                            expected: "bool".to_string(), 
                            got: x.get_type_name() 
                        },
                        loc: Some(condition.loc.clone())
                    });
                }
            };

            while cond  {
                let ret = run_statement(state, &body, program)?;

                if let StatementReturn::Return(v) = ret {
                    return Ok(StatementReturn::Return(v));
                }

                let eval_condition = match eval_expression(state, condition, program)? {
                    Ok(value) => value,
                    Err(value) => return Ok(StatementReturn::Return(value))
                };
                cond = match eval_condition {
                    Value::Bool(b) => b,
                    x => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError { 
                                expected: "bool".to_string(), 
                                got: x.get_type_name() 
                            },
                            loc: Some(condition.loc.clone())
                        });
                    }
                };
            }
        },
        ast::Stmt::Block { statements } => {
            match statements.as_slice() {
                [rest @ .., last] => {
                    
                    state.stack.new_frame();

                    for stmt in rest.iter() {
                        let ret = run_statement(state, stmt, program);
                        match ret {
                            Err(e) => {
                                return Err(e)
                            },
                            Ok(StatementReturn::Return(v)) => {
                                state.stack.drop_frame();                                
                                return Ok(StatementReturn::Return(v)) // we *return* a value
                            },
                            Ok(_) => {}
                        };
                    }

                    let ret = run_statement(state, last, program);
                    match ret {
                        Err(e) => {
                            return Err(e)
                        },
                        Ok(StatementReturn::Return(v)) => {
                            state.stack.drop_frame();                                
                            return Ok(StatementReturn::Return(v)) // we *return* a value
                        },
                        Ok(StatementReturn::Eval(v)) => {
                            state.stack.drop_frame();                                
                            return Ok(StatementReturn::Eval(v)) // we *return* a value
                        }
                        Ok(_) => {}
                    }
                },
                [] => return Ok(StatementReturn::None)
            }
        },
        ast::Stmt::Expression { expr: expression } => {
            let value = match eval_expression(state, expression, program)?{
                Ok(value) => value,
                Err(value) => return Ok(StatementReturn::Return(value))
            };
            return Ok(StatementReturn::Eval(value));
        }
    }

    return Ok(StatementReturn::None);
}


fn unpack_elements(state: &State, variables: &Vec<ast::LocExpr>, value: Value, value_loc: &ast::Loc) -> Result<Vec<(String, Value)>, InterpreterErrorMessage> {
    let values = match value.clone() {
        Value::Tuple(elements) => elements,
        Value::List(ptr) => {
            match state.heap.get(ptr) {
                Some(HeapObject::List(elements)) => elements.clone(),
                _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                            loc: Some(value_loc.clone())
                        }),
            }
        },
        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError { 
                                expected: "tuple or list".to_string(), 
                                got: value.get_type_name() 
                            },
                            loc: Some(value_loc.clone())
                        })
    };

    if variables.len() != values.len() {
        return Err(InterpreterErrorMessage {
                            error: InterpreterError::UnpackError(format!("Values mismatch: expected {} variables but got {} values", variables.len(), values.len())),
                            loc: Some(value_loc.clone())
                        })
    }

    let mut results: Vec<(String, Value)> = Vec::new();

    for (var, value) in variables.iter().zip(values) {
        match &var.expr {
            ast::Expr::Variable(var) => {
                results.push((var.clone(), value));
            },
            ast::Expr::Tuple(elements) | ast::Expr::List(elements) => {
                let rec = unpack_elements(state, elements, value, value_loc)?;
                results.extend(rec);
            },
            _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidAssignmentTarget,
                            loc: Some(var.loc.clone())
                        })
        }
    }

    Ok(results)
}


pub fn interpret(program: &ast::Program) -> Result<Value, InterpreterErrorMessage> {
    let mut state = State::new();

    let main_func = program.functions.get("main").unwrap();

    return Ok(run_statement(&mut state, &main_func.body, program)?.unwrap());
}

pub fn interpret_with_state(program: &ast::Program) -> Result<(Value, State), InterpreterErrorMessage> {
    let mut state = State::new();

    let main_func = program.functions.get("main").unwrap();

    return Ok((run_statement(&mut state, &main_func.body, program)?.unwrap(), state));
}