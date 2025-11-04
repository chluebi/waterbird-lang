use std::collections::{HashMap};
use std::hash::{Hash, Hasher};
use std::fmt;
use slab::Slab;

use crate::{ast};

pub type Ptr = usize;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(Ptr),
    Tuple(Vec<Value>),
    List(Ptr),
    Dictionary(Ptr),
    Lambda(Ptr),
    FunctionPtr(String)
}

impl Hash for Value {
    
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        match self {
            Value::Int(i) => i.hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Tuple(values) => values.iter().map(|v| v.hash(state)).collect(),
            Value::Str(ptr) => ptr.hash(state), // thanks to string interning!
            _ => unreachable!()
        }
    }

}

impl Value {

    pub fn hashable(&self) -> bool {
        match self {
            Value::Int(_) |
            Value::Bool(_) |
            Value::Str(_) => true,
            Value::Tuple(values) => values.iter().all(Value::hashable),
            _ => false
        }
    }

    pub fn get_type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dictionary(_) => "dict",
            Value::Lambda(_) => "lambda",
            Value::FunctionPtr(_) => "function",
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
    // For type mismatches, e.g., "Expected {expected}, got {got}"
    TypeError {
        expected: String,
        got: &'static str,
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
            InterpreterError::TypeError { expected, got } => {
                write!(f, "Type error: expected {}, got {}", expected, got)
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


#[derive(Debug)]
pub struct InterpreterErrorMessage {
    pub error: InterpreterError,
    pub loc: Option<ast::Loc>
}

pub fn eval_expression(state: &mut State, expression: &ast::LocExpr, program: &ast::Program) -> Result<Value, InterpreterErrorMessage> {
    
    match expression.expr {
        ast::Expr::Variable(ref v) => {
            match state.stack.get_value(&v) {
                Some(value) => Ok(value),
                _ => {
                    match program.functions.get(v) {
                        Some(_) => Ok(Value::FunctionPtr(v.clone())),
                        _ => Err(InterpreterErrorMessage {
                            error: InterpreterError::VariableNotFound(v.clone()),
                            loc: Some(expression.loc.clone())
                        })
                    }
                }
            }
        },
        ast::Expr::Int(ref i) => Ok(Value::Int(*i)),
        ast::Expr::Bool(ref b) => Ok(Value::Bool(*b)),
        ast::Expr::Str(ref s) => {
            let ptr = state.heap.intern_string(String::from(s));
            Ok(Value::Str(ptr))
        },
        ast::Expr::Tuple(ref values) => {
            let values: Result<Vec<Value>, InterpreterErrorMessage>
                = values.into_iter().map(|arg| eval_expression(state, &arg, program)).collect();
            let values: Vec<Value> = values?;
            Ok(Value::Tuple(values))
        },
        ast::Expr::List(ref values) => {
            let values: Result<Vec<Value>, InterpreterErrorMessage>
                = values.into_iter().map(|arg| eval_expression(state, &arg, program)).collect();
            let values: Vec<Value> = values?;
            let ptr = state.heap.alloc(HeapObject::List(values));
            Ok(Value::List(ptr))
        },
        ast::Expr::Dictionary(ref keys_values) => {
            let keys_values: Result<Vec<((Value, ast::Loc),(Value, ast::Loc))>, InterpreterErrorMessage>
                = keys_values.into_iter().map(|(key,value)| {
                    Ok(((eval_expression(state, &key, program)?, key.loc.clone()),
                    (eval_expression(state, &value, program)?, value.loc.clone())))
                }).collect();
            let keys_values: Vec<((Value, ast::Loc),(Value, ast::Loc))> = keys_values?;

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
            Ok(Value::Dictionary(ptr))
        },
        ast::Expr::BinOp { ref op, ref left, ref right } => {
            let left_value = eval_expression(state, &left, program)?;
            let right_value = eval_expression(state, &right, program)?;

            match (left_value.clone(), right_value.clone()) {
                (Value::Int(left_value), Value::Int(right_value)) => {
                    match op {
                        ast::BinOp::Eq => return Ok(Value::Bool(left_value == right_value)),
                        ast::BinOp::Neq => return Ok(Value::Bool(left_value != right_value)),
                        ast::BinOp::Leq => return Ok(Value::Bool(left_value <= right_value)),
                        ast::BinOp::Geq => return Ok(Value::Bool(left_value >= right_value)),
                        ast::BinOp::Lt => return Ok(Value::Bool(left_value < right_value)),
                        ast::BinOp::Gt => return Ok(Value::Bool(left_value > right_value)),
                        ast::BinOp::Add => return Ok(Value::Int(left_value + right_value)),
                        ast::BinOp::Sub => return Ok(Value::Int(left_value - right_value)),
                        ast::BinOp::Mul => return Ok(Value::Int(left_value * right_value)),
                        ast::BinOp::Div => return Ok(Value::Int(left_value / right_value)),
                        ast::BinOp::Mod => return Ok(Value::Int(left_value % right_value)),
                        ast::BinOp::ShiftLeft => return Ok(Value::Int(left_value << right_value)),
                        ast::BinOp::ShiftRightArith => return Ok(Value::Int(left_value >> right_value)),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { op: op.clone(), left: "int", right: "int" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                (Value::Bool(left_value), Value::Bool(right_value)) => {
                    match op {
                        ast::BinOp::Eq => return Ok(Value::Bool(left_value == right_value)),
                        ast::BinOp::Neq => return Ok(Value::Bool(left_value != right_value)),
                        ast::BinOp::And => return Ok(Value::Bool(left_value && right_value)),
                        ast::BinOp::Or => return Ok(Value::Bool(left_value || right_value)),
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
                            Ok(Value::List(ptr))
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
            let value = eval_expression(state, &expr, program)?;

            match value.clone() {
                Value::Int(value) => {
                    match op {
                        ast::UnOp::Neg => return Ok(Value::Int(-value)),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesUn { op: op.clone(), operand: "int" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                Value::Bool(value) => {
                    match op {
                        ast::UnOp::Not => return Ok(Value::Bool(!value)),
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
            let func_value = eval_expression(state, function, program)?;

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

                    let argument_values: Result<Vec<Value>, InterpreterErrorMessage>
                        = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expr, program)).collect();

                    let argument_values: Vec<Value> = argument_values?;

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
                    
                    let value: Result<Value, InterpreterErrorMessage> = eval_expression(state, &expr, program);

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
        },
        ast::Expr::Indexing { ref indexed, ref indexer } => {
            let original_indexed_value = eval_expression(state, &indexed, program)?;
            let original_indexer_value = eval_expression(state, &indexer, program)?;

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
                            Some(value) => return Ok(value.clone()),
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
                Value::Str(ptr) => {
                    match state.heap.get(ptr) {
                        Some(HeapObject::Str(str)) => {
                            let char_val = str.chars().nth(index).unwrap(); 
                            let char_str = char_val.to_string();
                            let new_ptr = state.heap.alloc(HeapObject::Str(char_str));
                            Ok(Value::Str(new_ptr))
                        },
                        _ => unreachable!() 
                    }
                },
                Value::Tuple(values) => {
                    Ok(values[index].clone())
                },
                Value::List(ptr) => {
                     match state.heap.get(ptr) {
                        Some(HeapObject::List(l)) => {
                            Ok(l[index].clone())
                        },
                        _ => unreachable!()
                     }
                },
                _ => unreachable!()
            }
        },
        ast::Expr::FunctionPtr(ref s) => {Ok(Value::FunctionPtr(s.clone()))},
        ast::Expr::Lambda { ref arguments, ref expr } => {
            let ptr = state.heap.alloc(HeapObject::Lambda {arguments: arguments.clone(), expr: expr.clone()});
            Ok(Value::Lambda(ptr))
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
                            Ok(Some(v)) => {
                                return Ok(v)
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
        Value::Str(ptr) => {
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

fn call_function(
    state: &mut State,
    function_name: &str,
    loc: &ast::Loc,
    positional_arguments: &Vec<ast::CallArgument>,
    variadic_argument: &Option<ast::CallArgument>,
    keyword_arguments: &Vec<ast::CallKeywordArgument>,
    keyword_variadic_argument: &Option<ast::CallArgument>,
    program: &ast::Program
) -> Result<Value, InterpreterErrorMessage> {
    let function = match program.functions.get(function_name) {
        Some(f) => f,
        _ => return Err(InterpreterErrorMessage {
                error: InterpreterError::FunctionNotFound(function_name.to_string()),
                loc: Some(loc.clone())
            })
    };

    let argument_values: Result<Vec<Value>, InterpreterErrorMessage>
        = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expr, program)).collect();

    let mut argument_values: Vec<Value> = argument_values?;

    match &variadic_argument {
        Some(arg) => {
            let value = eval_expression(state, &arg.expr, program)?;
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

    let keyword_values: Result<HashMap<String, (Option<&ast::CallKeywordArgument>, Value)>, InterpreterErrorMessage> = keyword_arguments.iter()
        .map(|arg| {
            eval_expression(state, &arg.expr, program).map(|value| (arg.name.clone(), (Some(arg), value)))
        })
        .collect();

    let mut keyword_values: HashMap<String, (Option<&ast::CallKeywordArgument>, Value)> = keyword_values?;

    match keyword_variadic_argument {
        Some(arg) => {
            let value = eval_expression(state, &arg.expr, program)?;
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
                            Value::Str(ptr) => {
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

    if argument_values.len() < function.contract.positional_arguments.len() {
        let pos = argument_values.len();
        let missing_arg = function.contract.positional_arguments.get(pos).unwrap();
        return Err(InterpreterErrorMessage {
            error: InterpreterError::ArgumentError(format!("Missing argument: '{}'", missing_arg.name)),
            loc: Some(missing_arg.loc.clone())
        })
    }

    if function.contract.variadic_argument.is_none() && argument_values.len() > function.contract.positional_arguments.len() {
        let pos = function.contract.positional_arguments.len();
        
        let extra_arg_expression = match pos < positional_arguments.len() {
            true => positional_arguments.get(pos).unwrap(),
            false => &variadic_argument.clone().unwrap()
        };

        return Err(InterpreterErrorMessage {
            error: InterpreterError::ArgumentError("Unexpected positional argument".to_string()),
            loc: Some(extra_arg_expression.loc.clone())
        })
    }

    let mut new_values: HashMap<String, Value> = function.contract.positional_arguments.iter().zip(argument_values.iter()).map(|(arg, value)| (arg.name.clone(), value.clone())).collect();

    if argument_values.len() > function.contract.positional_arguments.len() {
        // from previous logic the only way this happens if we have a variadic accepting argument in the function
        let extra_args = &argument_values[function.contract.positional_arguments.len()..];
        let ptr = state.heap.alloc(HeapObject::List(extra_args.to_vec()));

        new_values.insert(function.contract.variadic_argument.clone().unwrap().name, Value::List(ptr));
    }

    let mut keyword_variadic_arguments: HashMap<Value, Value> = HashMap::new();

    for keyword_arg in function.contract.keyword_arguments.clone() {
        new_values.insert(keyword_arg.name, eval_expression(state, &keyword_arg.expr, program)?);
    }

    for (key, (arg, value)) in keyword_values {
        match function.contract.keyword_arguments.iter().filter(|y| y.name == key).peekable().peek() {
            Some(_) => {
                new_values.insert(key.clone(), value);
            }
            _ => {
                match &function.contract.keyword_variadic_argument {
                    Some(_) => {
                        let ptr = state.heap.alloc(HeapObject::Str(String::from(key)));
                        keyword_variadic_arguments.insert(Value::Str(ptr), value);
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

    match &function.contract.keyword_variadic_argument {
        Some(arg) => {
            let ptr = state.heap.alloc(HeapObject::Dictionary(keyword_variadic_arguments));
            new_values.insert(arg.name.clone(), Value::Dictionary(ptr) );
        },
        _ => assert!(keyword_variadic_arguments.is_empty())
    }



    state.stack.new_function_context();

    new_values.into_iter().for_each(|(n,v)| {state.stack.update_variable(&n, v);});

    let value = run_statement(state, &function.body, program);

    state.stack.drop_function_context();

    match value {
        Ok(Some(v)) => {
            Ok(v)
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


pub fn run_statement(state: &mut State, stmt: &ast::LocStmt, program: &ast::Program) -> Result<Option<Value>, InterpreterErrorMessage> {
    match &stmt.stmt {
        ast::Stmt::Assignment { target, expr: expression } => {
            match &target.expr {
                ast::Expr::Variable(v) => {
                    let value = eval_expression(state, expression, program)?;
                    state.stack.update_variable(&v,value);
                    return Ok(None)
                },
                ast::Expr::Indexing { indexed, indexer } => {
                    let original_indexed_value = eval_expression(state, &indexed, program)?;
                    let original_indexer_value = eval_expression(state, &indexer, program)?;

                    let value = eval_expression(state, expression, program)?;

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

                        return Ok(None);
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
                        Value::Str(_) => {
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
                    let value = eval_expression(state, expression, program)?;
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
            let value = eval_expression(state, expression, program)?;
            return Ok(Some(value));
        },
        ast::Stmt::IfElse { cond: condition, if_body, else_body } => {
            let eval_condition = eval_expression(state, &condition, program)?;

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
            let eval_condition = eval_expression(state, condition, program)?;

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
                if let Some(v) = ret {
                    return Ok(Some(v));
                }

                let eval_condition = eval_expression(state, condition, program)?;
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
            state.stack.new_frame();

            for stmt in statements.iter() {
                let ret = run_statement(state, stmt, program);
                match ret {
                    Err(e) => {
                        return Err(e)
                    },
                    Ok(Some(v)) => {
                        return Ok(Some(v))
                    },
                    Ok(_) => {}
                };
            }

            state.stack.drop_frame();
        },
        ast::Stmt::Expression { expr: expression } => {
            // Behave like return statement
            return Ok(Some(eval_expression(state, expression, program)?));
        }
    }

    return Ok(None);
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