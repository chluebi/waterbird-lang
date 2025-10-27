use std::collections::{HashMap};
use std::hash::{Hash, Hasher};
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


#[derive(Debug)]
pub enum InterpreterError {
    Panic(String)
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
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::Panic(String::from("Variable not found ") + v),
                    loc: Some(expression.loc.clone())
                })
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
                        error: InterpreterError::Panic(String::from("Unhashable key")),
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

            match (left_value, right_value) {
                (Value::Int(left_value), Value::Int(right_value)) => {
                    match op {
                        ast::BinOp::Add => return Ok(Value::Int(left_value + right_value)),
                        ast::BinOp::Sub => return Ok(Value::Int(left_value - right_value))
                    }
                },

                (Value::Bool(left_value), Value::Bool(right_value)) => {
                    match op {
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Invalid type for operator")),
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Invalid type for operator")),
                            loc: Some(expression.loc.clone())
                        })
            }
        },
        ast::Expr::Unop { ref op, ref expr } => {
            let value = eval_expression(state, &expr, program)?;

            match value {
                Value::Int(value) => {
                    match op {
                        ast::UnOp::Neg => return Ok(Value::Int(-value))
                    }
                },

                _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Invalid type for operator")),
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
            match eval_expression(state, function, program)? {
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
                            error: InterpreterError::Panic(String::from("Expected Lambda")),
                            loc: Some(function.loc.clone())
                        })
                    };

                    let argument_values: Result<Vec<Value>, InterpreterErrorMessage>
                        = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expression, program)).collect();

                    let argument_values: Vec<Value> = argument_values?;

                    if argument_values.len() < arguments.len() {
                        let pos = argument_values.len();
                        let missing_arg = arguments.get(pos).unwrap();
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Missing argument")),
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
                            error: InterpreterError::Panic(String::from("Unexpected argument")),
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
                        error: InterpreterError::Panic(String::from("Function not found")),
                        loc: Some(expression.loc.clone())
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
                                error: InterpreterError::Panic(String::from("Unhashable key for dictionary lookup")),
                                loc: Some(indexer.loc.clone())
                            })
                        }
                        match dict.get(&original_indexer_value) {
                            Some(value) => return Ok(value.clone()),
                            _ => return Err(InterpreterErrorMessage {
                                error: InterpreterError::Panic(String::from("Key not found")),
                                loc: Some(indexer.loc.clone())
                            })
                        }
                    },
                    _ => {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Expected Dictionary Heap Object")),
                            loc: Some(indexed.loc.clone())
                        })
                    }
                }
            }

            let mut indexer_value = match original_indexer_value {
                Value::Int(i) => i,
                _ => return Err(InterpreterErrorMessage {
                    error: InterpreterError::Panic(String::from("Indexer for strings, tuples, lists needs to be an int")),
                    loc: Some(indexer.loc.clone())
                })
            };

            let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;

            if indexer_value < 0 {
                indexer_value = (indexed_length as i64) + indexer_value;
            }
            

            if indexer_value < 0 || indexer_value >= (indexed_length as i64) {
                 return Err(InterpreterErrorMessage {
                    error: InterpreterError::Panic(String::from("Index out of bounds")),
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
            todo!()
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
                        error: InterpreterError::Panic(String::from("Expected String Heap Object")),
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
                        error: InterpreterError::Panic(String::from("Expected List Heap Object")),
                        loc: Some(indexed.loc.clone())
                    })
                }
            }
        },
        _ => return Err(InterpreterErrorMessage {
            error: InterpreterError::Panic(String::from("Only strings, tuples, lists and dictionaries can be indexed")),
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
                error: InterpreterError::Panic(String::from("Function not found")),
                loc: Some(loc.clone())
            })
    };

    let argument_values: Result<Vec<Value>, InterpreterErrorMessage>
        = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expression, program)).collect();

    let mut argument_values: Vec<Value> = argument_values?;

    match &variadic_argument {
        Some(arg) => {
            let value = eval_expression(state, &arg.expression, program)?;
            let extra_args: Vec<Value> = match value {
                Value::Tuple(elements) => elements,
                Value::List(ptr) => {
                    match state.heap.get(ptr) {
                        Some(HeapObject::List(list)) => list.clone(),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Expected List object")),
                            loc: Some(arg.loc.clone())
                        })
                    }
                },
                x => return Err(InterpreterErrorMessage {
                    error: InterpreterError::Panic("Only tuples or list can be passed as variadic arguments".to_string()),
                    loc: Some(arg.loc.clone())
                })
            };
            argument_values.extend(extra_args);
        },
        _ => ()
    }

    let keyword_values: Result<HashMap<String, (Option<&ast::CallKeywordArgument>, Value)>, InterpreterErrorMessage> = keyword_arguments.iter()
        .map(|arg| {
            eval_expression(state, &arg.expression, program).map(|value| (arg.name.clone(), (Some(arg), value)))
        })
        .collect();

    let mut keyword_values: HashMap<String, (Option<&ast::CallKeywordArgument>, Value)> = keyword_values?;

    match keyword_variadic_argument {
        Some(arg) => {
            let value = eval_expression(state, &arg.expression, program)?;
            match value {
                Value::Dictionary(ptr) => {
                    let index_ref = match state.heap.get(ptr) {
                        Some(HeapObject::Dictionary(d)) => d,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Expected Dict object")),
                            loc: Some(arg.loc.clone())
                        })
                    };

                    for (key, value) in index_ref.iter() {
                        match key {
                            Value::Str(ptr) => {
                                let s = match state.heap.get(*ptr) {
                                    Some(HeapObject::Str(s)) => s,
                                    _ => return Err(InterpreterErrorMessage {
                                        error: InterpreterError::Panic(String::from("Expected Str object")),
                                        loc: Some(arg.loc.clone())
                                    })
                                };

                                if !keyword_values.contains_key(&s.clone()) {
                                    keyword_values.insert(s.clone(), (None, value.clone()));
                                }
                            },
                            x => return Err(InterpreterErrorMessage {
                                error: InterpreterError::Panic(String::from("Dicts passed as variadic kwargs have to only have string keys")),
                                loc: Some(arg.loc.clone())
                            })                               
                        }
                    }
                },
                x => return Err(InterpreterErrorMessage {
                    error: InterpreterError::Panic("Only dict can be passed as keyword variadic arguments".to_string()),
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
            error: InterpreterError::Panic(String::from("Missing argument")),
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
            error: InterpreterError::Panic(String::from("Unexpected argument")),
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
        new_values.insert(keyword_arg.name, eval_expression(state, &keyword_arg.expression, program)?);
    }

    for (key, (arg, value)) in keyword_values {
        match function.contract.keyword_arguments.iter().filter(|y| y.name == key).peekable().peek() {
            Some(_) => {
                new_values.insert(key, value);
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
                                error: InterpreterError::Panic(String::from("Unexpected Argument")),
                                loc: Some(arg.loc.clone())
                            })
                        },
                        _ => {
                            // If we do not have an arg passed this means that it originally came from the keyword_variadic argument
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::Panic(String::from("Unexpected Argument")),
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
                error: InterpreterError::Panic(String::from("Function should have returned a value")),
                loc: Some(loc.clone())
            })
        },
        Err(e) => return Err(e)
    }
}


pub fn run_statement(state: &mut State, stmt: &ast::LocStmt, program: &ast::Program) -> Result<Option<Value>, InterpreterErrorMessage> {
    match &stmt.stmt {
        ast::Stmt::Assignment { target, expression } => {
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
                                error: InterpreterError::Panic(String::from("Unhashable key for dictionary assignment")),
                                loc: Some(indexer.loc.clone())
                            })
                        }
                        match state.heap.get_mut(ptr) {
                            Some(HeapObject::Dictionary(dict)) => {
                                dict.insert(original_indexer_value, value);
                            },
                            _ => {
                                return Err(InterpreterErrorMessage {
                                    error: InterpreterError::Panic(String::from("Expected Dictionary Heap Object")),
                                    loc: Some(indexed.loc.clone())
                                })
                            }
                        }

                        return Ok(None);
                    }

                    let mut indexer_value = match original_indexer_value {
                        Value::Int(i) => i,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Indexer for strings, tuples, lists needs to be an int")),
                            loc: Some(indexer.loc.clone())
                        })
                    };

                    let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;

                    if indexer_value < 0 {
                        indexer_value = (indexed_length as i64) + indexer_value;
                    }
                    

                    if indexer_value < 0 || indexer_value >= (indexed_length as i64) {
                        return Err(InterpreterErrorMessage {
                            error: InterpreterError::Panic(String::from("Index out of bounds")),
                            loc: Some(indexer.loc.clone())
                        });
                    }

                    let index = indexer_value as usize;

                    match original_indexed_value {
                        Value::Str(ptr) => {
                             return Err(InterpreterErrorMessage {
                                error: InterpreterError::Panic(String::from("Strings are immutable")),
                                loc: Some(indexer.loc.clone())
                            });
                        },
                        Value::Tuple(values) => {
                            return Err(InterpreterErrorMessage {
                                error: InterpreterError::Panic(String::from("Tuples are immutable")),
                                loc: Some(indexer.loc.clone())
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
                _ => todo!()
            }
        },
        ast::Stmt::FunctionCall { expression } => {eval_expression(state, expression, program)?;},
        ast::Stmt::Return { expression } => {
            let value = eval_expression(state, expression, program)?;
            return Ok(Some(value));
        },
        ast::Stmt::IfElse { condition, if_body, else_body } => {
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
                        error: InterpreterError::Panic(format!("Using non-bool value in if condition")),
                        loc: Some(condition.loc.clone())
                    });
                }
            };
        },
        ast::Stmt::While { condition, body } => {
            let eval_condition = eval_expression(state, condition, program)?;

            let mut cond = match eval_condition {
                Value::Bool(b) => b,
                x => {
                    return Err(InterpreterErrorMessage {
                        error: InterpreterError::Panic(format!("Using non-bool value in loop")),
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
                            error: InterpreterError::Panic(format!("Using non-bool value in loop")),
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
        }
    }

    return Ok(None);
}
