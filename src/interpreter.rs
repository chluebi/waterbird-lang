use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::fs::File;
use std::io::{self, Read};
use std::fmt;
use slab::Slab;

use crate::{ast};

pub type Ptr = usize;

macro_rules! eval_or_return_from_expr {
    ($state:expr, $expression:expr, $program:expr) => {
        match eval_expression($state, $expression, $program)? {
            Ok(value) => value,
            Err(value) => return Ok(Err(value)),
        }
    };
}

macro_rules! eval_or_return_from_stmt {
    ($state:expr, $expression:expr, $program:expr) => {
        match eval_expression($state, $expression, $program)? {
            Ok(value) => value,
            Err(value) => return Ok(StatementReturn::Return(value)),
        }
    };
}



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

    pub fn truthy(&self, state: &State, loc: &ast::Loc) -> Result<bool, InterpreterErrorMessage> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Int(i) => Ok(*i != 0),
            Value::Void => Ok(false),
            Value::String(ptr) => {
                let s = state.heap.get_string(*ptr, Some(&loc))?;
                Ok(!s.is_empty())
            },
            Value::Tuple(v) => Ok(!v.is_empty()),
            Value::List(ptr) => {
                // Retrieve list content from the heap
                let l = state.heap.get_list(*ptr, Some(&loc))?;
                Ok(!l.is_empty())
            },
            Value::Dictionary(ptr) => {
                let d = state.heap.get_dict(*ptr, Some(&loc))?;
                Ok(!d.is_empty())
            },
            // Functions and Lambdas are generally truthy
            Value::Lambda(_) | Value::FunctionPtr(_) | Value::NameSpacePtr(_) => Ok(true),
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
       expr: Box<ast::LocExpr>,
       captured: HashMap<String, Value>
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

    pub fn _get(&self, ptr: Ptr) -> Option<&HeapObject> {
        self.objects.get(ptr)
    }

    pub fn get_string(&self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<&String, InterpreterErrorMessage> {
        match self.objects.get(ptr) {
            Some(HeapObject::Str(str)) => Ok(str),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected String Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
    }

    pub fn get_list(&self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<&Vec<Value>, InterpreterErrorMessage> {
        match self.objects.get(ptr) {
            Some(HeapObject::List(l)) => Ok(l),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
    }

    pub fn get_dict(&self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<&HashMap<Value, Value>, InterpreterErrorMessage> {
        match self.objects.get(ptr) {
            Some(HeapObject::Dictionary(d)) => Ok(d),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
    }

    pub fn get_lambda(&self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<(&Vec<ast::LambdaArgument>, &ast::LocExpr, &HashMap<String, Value>), InterpreterErrorMessage> {
        match self.objects.get(ptr) {
            Some(HeapObject::Lambda {arguments, expr, captured}) => Ok((arguments, *&expr, captured)),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected Lambda Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
    }

    pub fn get_list_mut(&mut self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<&mut Vec<Value>, InterpreterErrorMessage> {
        match self.objects.get_mut(ptr) {
            Some(HeapObject::List(l)) => Ok(l),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected List Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
    }

    pub fn get_dict_mut(&mut self, ptr: Ptr, loc: Option<&ast::Loc>) -> Result<&mut HashMap<Value, Value>, InterpreterErrorMessage> {
        match self.objects.get_mut(ptr) {
            Some(HeapObject::Dictionary(d)) => Ok(d),
            _ => Err(InterpreterErrorMessage {
                error: InterpreterError::InternalError("Expected Dictionary Heap Object".to_string()),
                loc: loc.cloned()
            })
        }
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
                match self.heap.get_string(*ptr, None) {
                    Ok(s) => {
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
                match self.heap.get_list(*ptr, None) {
                    Ok(l) => {
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
                match self.heap.get_dict(*ptr, None) {
                    Ok(d) => {
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

    pub fn shadow_variable(&mut self, name: &str, value: Value) -> Option<Value> {
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
    FileError(String), // for the read internal function
    AssertionError(String),
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
            InterpreterError::FileError(msg ) => write!(f, "File error: {}", msg),
            InterpreterError::AssertionError(msg ) => write!(f, "Assertion error: {}", msg),
            InterpreterError::InternalError(msg) => write!(f, "Interpreter internal error: {}. This is likely a bug.", msg),
        }
    }
}


#[derive(Debug, PartialEq)]
pub struct InterpreterErrorMessage {
    pub error: InterpreterError,
    pub loc: Option<ast::Loc>
}

fn deep_equals(left: &Value, right: &Value, heap: &Heap) -> bool {
    match (left, right) {
        (Value::Void, Value::Void) => true,
        (Value::Int(l), Value::Int(r)) => l == r,
        (Value::Bool(l), Value::Bool(r)) => l == r,
        (Value::String(l_ptr), Value::String(r_ptr)) => l_ptr == r_ptr, // Interned
        (Value::FunctionPtr(l_name), Value::FunctionPtr(r_name)) => l_name == r_name,
        (Value::NameSpacePtr(l_name), Value::NameSpacePtr(r_name)) => l_name == r_name,
        (Value::Lambda(l_ptr), Value::Lambda(r_ptr)) => l_ptr == r_ptr, // Reference equality

        (Value::Tuple(l_vec), Value::Tuple(r_vec)) => {
            if l_vec.len() != r_vec.len() {
                return false;
            }
            l_vec.iter().zip(r_vec.iter()).all(|(l, r)| deep_equals(l, r, heap))
        },

        (Value::List(l_ptr), Value::List(r_ptr)) => {
            if l_ptr == r_ptr { return true; }

            let l_obj = heap.get_list(*l_ptr, None);
            let r_obj = heap.get_list(*r_ptr, None);

            match (l_obj, r_obj) {
                (Ok(l_vec), Ok(r_vec)) => {
                    if l_vec.len() != r_vec.len() {
                        return false;
                    }
                    l_vec.iter().zip(r_vec.iter()).all(|(l, r)| deep_equals(l, r, heap))
                },
                _ => false
            }
        },

        (Value::Dictionary(l_ptr), Value::Dictionary(r_ptr)) => {
            if l_ptr == r_ptr { return true; } // Same object

            let l_obj = heap.get_dict(*l_ptr, None);
            let r_obj = heap.get_dict(*r_ptr, None);

            match (l_obj, r_obj) {
                (Ok(l_map), Ok(r_map)) => {
                    if l_map.len() != r_map.len() {
                        return false;
                    }
                    
                    l_map.iter().all(|(l_key, l_val)| {
                        r_map.get(l_key)
                             .map_or(false, |r_val| deep_equals(l_val, r_val, heap))
                    })
                },
                _ => false
            }
        },

        _ => false
    }
}


fn resolve_variable_from_state(state: &mut State, v: &str, program: &ast::Program) -> Result<Option<Value>, InterpreterErrorMessage> {
    match state.stack.get_value(&v) {
        Some(value) => return Ok(Some(value)),
        _ => ()
    }

    match program.functions.get(v) {
        Some(_) => return Ok(Some(Value::FunctionPtr(v.to_string()))),
        _ => ()
    }

    match v {
        "Int" => return Ok(Some(Value::NameSpacePtr(String::from("Int")))),
        "Bool" => return Ok(Some(Value::NameSpacePtr(String::from("Bool")))),
        "String" => return Ok(Some(Value::NameSpacePtr(String::from("String")))),
        "Tuple" => return Ok(Some(Value::NameSpacePtr(String::from("Tuple")))),
        "List" => return Ok(Some(Value::NameSpacePtr(String::from("List")))),
        "Dict" => return Ok(Some(Value::NameSpacePtr(String::from("Dict")))),
        _ => ()
    }

    match v {
        "int" => return Ok(Some(Value::FunctionPtr(String::from("int")))),
        "bool" => return Ok(Some(Value::FunctionPtr(String::from("bool")))),
        "str" => return Ok(Some(Value::FunctionPtr(String::from("str")))),
        "tuple" => return Ok(Some(Value::FunctionPtr(String::from("tuple")))),
        "list" => return Ok(Some(Value::FunctionPtr(String::from("list")))),
        "dict" => return Ok(Some(Value::FunctionPtr(String::from("dict")))),

        "clone" => return Ok(Some(Value::FunctionPtr(String::from("clone")))),
        "len" => return Ok(Some(Value::FunctionPtr(String::from("len")))),
        "print" => return Ok(Some(Value::FunctionPtr(String::from("print")))),
        "read" => return Ok(Some(Value::FunctionPtr(String::from("read")))),
        "read_as_list" => return Ok(Some(Value::FunctionPtr(String::from("read_as_list")))),

        "split" => return Ok(Some(Value::FunctionPtr(String::from("split")))),
        "range" => return Ok(Some(Value::FunctionPtr(String::from("range")))),

        "assert" => return Ok(Some(Value::FunctionPtr(String::from("assert")))),
        "dealloc" => return Ok(Some(Value::FunctionPtr(String::from("dealloc")))),

        _ => ()
    }

    Ok(None)
}


impl ast::LocExpr {

    fn target_defined_variables(self: &Self) -> HashSet<String> {
        match &self.expr {
            ast::Expr::Variable(v) => vec![v.clone()].into_iter().collect(),
            ast::Expr::Tuple(vars) | ast::Expr::List(vars) => vars.iter().flat_map(ast::LocExpr::target_defined_variables).collect(),
            _ => HashSet::new()
        }
    }

    fn target_used_variables(self: &Self) -> HashSet<String> {
        match &self.expr {
            ast::Expr::Variable(_) => HashSet::new(),
            ast::Expr::Indexing {indexed, indexer } => {
                let mut ret = ast::LocExpr::free_variables(indexed);
                ret.extend(ast::LocExpr::free_variables(indexer));
                ret
            },
            ast::Expr::DotAccess(expr, _) => ast::LocExpr::free_variables(expr),
            ast::Expr::Tuple(vars) | ast::Expr::List(vars) => {
                vars.iter().flat_map(ast::LocExpr::target_used_variables).collect()
            },
            _ => unreachable!()
        }
    }

    fn free_variables(self: &Self) -> HashSet<String> {
        match self.expr {
            ast::Expr::Variable(ref v) => {
                vec![v.clone()].into_iter().collect()
            }
            ast::Expr::DotAccess(ref expr, _) => ast::LocExpr::free_variables(&expr),
            ast::Expr::Int(_) => HashSet::new(),
            ast::Expr::Bool(_) => HashSet::new(),
            ast::Expr::Str(_) => HashSet::new(),
            ast::Expr::Tuple(ref elts) => elts.iter().flat_map(ast::LocExpr::free_variables).collect(),
            ast::Expr::List(ref l) => l.iter().flat_map(ast::LocExpr::free_variables).collect(),
            ast::Expr::Dictionary(ref d) => d.iter().flat_map(|(k,v)| {let mut k = ast::LocExpr::free_variables(k); k.extend(ast::LocExpr::free_variables(v)); k}).collect(),
            ast::Expr::BinOp { op: _, ref left, ref right} => {let mut left = ast::LocExpr::free_variables(&left); left.extend(ast::LocExpr::free_variables(&right)); left},
            ast::Expr::UnOp { op: _, ref expr } => ast::LocExpr::free_variables(&expr),
            ast::Expr::FunctionCall { ref function, ref positional_arguments, ref variadic_argument, ref keyword_arguments, ref keyword_variadic_argument } => {
                let mut ret = ast::LocExpr::free_variables(&function);
                ret.extend(positional_arguments.iter().flat_map(|arg| ast::LocExpr::free_variables(&arg.expr)));
                if let Some(variadic_argument) = variadic_argument {
                    ret.extend(ast::LocExpr::free_variables(&variadic_argument.expr));
                }
                ret.extend(keyword_arguments.iter().flat_map(|arg| ast::LocExpr::free_variables(&arg.expr)));
                if let Some(keyword_variadic_argument) = keyword_variadic_argument {
                    ret.extend(ast::LocExpr::free_variables(&keyword_variadic_argument.expr));
                }
                ret
            },
            ast::Expr::Indexing { ref indexed, ref indexer } => {let mut indexed = ast::LocExpr::free_variables(&indexed); indexed.extend(ast::LocExpr::free_variables(&indexer)); indexed}
            ast::Expr::Slice { ref indexed, ref indexer_start, ref indexer_border, ref indexer_step } => {
                let mut ret = ast::LocExpr::free_variables(&indexed);
                if let Some(indexer_start) = indexer_start {
                    ret.extend(ast::LocExpr::free_variables(&indexer_start));
                }
                if let Some(indexer_border) = indexer_border {
                    ret.extend(ast::LocExpr::free_variables(&indexer_border));
                }
                if let Some(indexer_step) = indexer_step {
                    ret.extend(ast::LocExpr::free_variables(&indexer_step));
                }
                ret
            },
            ast::Expr::FunctionPtr(_) => HashSet::new(),
            ast::Expr::Lambda { ref arguments, ref expr } => {
                let mut ret = HashSet::new();
                for v in ast::LocExpr::free_variables(expr) {
                    if !arguments.iter().any(|arg| arg.name == v) {
                        ret.insert(v);
                    }
                }
                ret
            },
            ast::Expr::Block { ref statements } => {
                let mut ret = HashSet::new();
                let mut defined_in_scope = HashSet::new();

                for stmt in statements {
                    let stmt_free = ast::LocStmt::free_variables(stmt);

                    for var in stmt_free {
                        if !defined_in_scope.contains(&var) {
                            ret.insert(var);
                        }
                    }

                    let stmt_defined = ast::LocStmt::defined_variables(stmt);
                    defined_in_scope.extend(stmt_defined);
                }
                ret
            }
        }
    }
    
}


impl ast::LocStmt {

    fn defined_variables(self: &Self) -> HashSet<String> {
        match &self.stmt {
            ast::Stmt::Assignment { target, .. } => target.target_defined_variables(),
            _ => HashSet::new()
        }
    }

    fn free_variables(self: &Self) -> HashSet<String> {
        match self.stmt {
            ast::Stmt::Assignment { ref target, ref expr } => {
                let mut ret = target.target_used_variables();
                ret.extend(ast::LocExpr::free_variables(&expr));
                ret
            },
            ast::Stmt::FunctionCall { ref expr } => ast::LocExpr::free_variables(&expr),
            ast::Stmt::Return { ref expr } => ast::LocExpr::free_variables(&expr),
            ast::Stmt::IfElse { ref cond, ref if_body, ref else_body } => {
                let mut ret = ast::LocExpr::free_variables(cond);
                ret.extend(ast::LocStmt::free_variables(if_body));
                ret.extend(ast::LocStmt::free_variables(else_body));
                ret
            }, 
            ast::Stmt::While { ref cond, ref body } => {
                let mut ret = ast::LocExpr::free_variables(cond);
                ret.extend(ast::LocStmt::free_variables(body));
                ret
            },
            ast::Stmt::Block { ref statements } => {
                let mut ret = HashSet::new();
                let mut defined_in_scope = HashSet::new();

                for stmt in statements {
                    let stmt_free = ast::LocStmt::free_variables(stmt);

                    for var in stmt_free {
                        if !defined_in_scope.contains(&var) {
                            ret.insert(var);
                        }
                    }

                    let stmt_defined = ast::LocStmt::defined_variables(stmt);
                    defined_in_scope.extend(stmt_defined);
                }
                ret
            },
            ast::Stmt::Expression { ref expr } => ast::LocExpr::free_variables(&expr),
        }
    }

}


// we return Ok(Ok(Value)) if we just evaluate
// we short-circuit with Ok(Err(Value)) as this means we have a direct return value
pub fn eval_expression(state: &mut State, expression: &ast::LocExpr, program: &ast::Program) -> Result<Result<Value, Value>, InterpreterErrorMessage> {
    
    match expression.expr {
        ast::Expr::Variable(ref v) => {
            if let Some(v) = resolve_variable_from_state(state, v, program)? {
                return Ok(Ok(v));
            }

            Err(InterpreterErrorMessage {
                error: InterpreterError::VariableNotFound(v.clone()),
                loc: Some(expression.loc.clone())
            })
        },
        ast::Expr::DotAccess(ref e, ref v) => {
            let value = eval_or_return_from_expr!(state, e, program);

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
            match op {
                // short-circuiting
                ast::BinOp::And | ast::BinOp::Or => {
                    let left_value = eval_or_return_from_expr!(state, left, program);

                    let left_bool = match left_value {
                        Value::Bool(b) => b,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError {
                                expected: "bool".to_string(),
                                got: left_value.get_type_name()
                            },
                            loc: Some(left.loc.clone())
                        })
                    };

                    if *op == ast::BinOp::And && !left_bool {
                        return Ok(Ok(Value::Bool(false)));
                    }
                    if *op == ast::BinOp::Or && left_bool {
                        return Ok(Ok(Value::Bool(true)));
                    }

                    let right_value = eval_or_return_from_expr!(state, right, program);

                    match right_value {
                        Value::Bool(_) => return Ok(Ok(right_value)),
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError {
                                expected: "bool".to_string(),
                                got: right_value.get_type_name()
                            },
                            loc: Some(right.loc.clone())
                        })
                    }
                },
                _ => ()
            }

            let left_value = eval_or_return_from_expr!(state, left, program);
            let right_value = eval_or_return_from_expr!(state, right, program);
            let left_value_type_name = left_value.get_type_name();
            let right_value_type_name = right_value.get_type_name();

            // Handle Eq and Neq using deep_equals
            match op {
                ast::BinOp::Eq => {
                    return Ok(Ok(Value::Bool(deep_equals(&left_value, &right_value, &state.heap))));
                },
                ast::BinOp::Neq => {
                    return Ok(Ok(Value::Bool(!deep_equals(&left_value, &right_value, &state.heap))));
                },
                ast::BinOp::In => {
                    match (left_value, right_value) {
                        (needle, Value::Tuple(values)) => {
                            return Ok(Ok(Value::Bool(values.iter().any(|val| val == &needle))));
                        },
                        (needle, Value::List(ptr)) => {
                            return Ok(Ok(Value::Bool(state.heap.get_list(ptr, Some(&right.loc))?.iter().any(|val| val == &needle))))
                        },
                        (needle, Value::Dictionary(ptr)) => {
                            return Ok(Ok(Value::Bool(state.heap.get_dict(ptr, Some(&right.loc))?.contains_key(&needle))))
                        },
                        (Value::String(needle_ptr), Value::String(haystack_ptr)) => {
                            let (needle, haystack) = (state.heap.get_string(needle_ptr, Some(&right.loc))?, state.heap.get_string(haystack_ptr, Some(&left.loc))?);
                            return Ok(Ok(Value::Bool(haystack.contains(needle))))
                        },
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { 
                                op: op.clone(), 
                                left: left_value_type_name, 
                                right: right_value_type_name
                            },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },
                _ => ()
            }

            match (left_value, right_value) {
                (Value::Int(left_value), Value::Int(right_value)) => {
                    match op {
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

                (Value::List(left_ptr), Value::List(right_ptr)) => {
                    let left_value = state.heap.get_list(left_ptr, Some(&left.loc))?;
                    let right_value = state.heap.get_list(right_ptr, Some(&right.loc))?;

                    match op {
                        ast::BinOp::Add => {
                            let mut new_list = left_value.clone();
                            new_list.extend(right_value.clone());
                            let ptr = state.heap.alloc(HeapObject::List(new_list));
                            Ok(Ok(Value::List(ptr)))
                        }
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::InvalidOperandTypesBin { op: op.clone(), left: "list", right: "list" },
                            loc: Some(expression.loc.clone())
                        })
                    }
                },

                (Value::String(left_ptr), Value::String(right_ptr)) => {
                    let left_value = state.heap.get_string(left_ptr, Some(&left.loc))?;
                    let right_value = state.heap.get_string(right_ptr, Some(&right.loc))?;

                    match op {
                        ast::BinOp::Add => {
                            let ptr = state.heap.intern_string(format!("{}{}", left_value, right_value));
                            Ok(Ok(Value::String(ptr)))
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
                                left: left_value_type_name, 
                                right: right_value_type_name 
                            },
                            loc: Some(expression.loc.clone())
                        })
            }
        },
        ast::Expr::UnOp { ref op, ref expr } => {
            let value = eval_or_return_from_expr!(state, expr, program);

            match value {
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
                let self_value = eval_or_return_from_expr!(state, base_expr, program);

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
                let func_value = eval_or_return_from_expr!(state, function, program);

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
                        let argument_values: Result<Result<Vec<Value>, Value>, InterpreterErrorMessage>
                            = positional_arguments.iter().map(|arg| eval_expression(state, &arg.expr, program)).collect();

                        let argument_values: Vec<Value> = match argument_values? {
                            Ok(values) => values,
                            Err(value) => return Ok(Err(value))
                        };

                        let (arguments, expr, captured) = match state.heap.get_lambda(ptr, Some(&function.loc))? {
                            (arguments, expr, captured) => (arguments.clone(), expr.clone(), captured.clone())
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
                        
                        let _ = captured.into_iter().for_each(|(n, v)| {state.stack.shadow_variable(&n, v);});
                        let _ = new_values.into_iter().for_each(|(n,v)| {state.stack.shadow_variable(&n, v);});
                        
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
            let original_indexed_value = eval_or_return_from_expr!(state, indexed, program);
            let original_indexer_value = eval_or_return_from_expr!(state, indexer, program);

            if let Value::Dictionary(ptr) = original_indexed_value {
                let dict = state.heap.get_dict(ptr, Some(&indexed.loc))?;

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
            }

            let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;
            let index = wrap_index(original_indexer_value, indexer.loc.clone(), indexed_length)?;
            

            match original_indexed_value {
                Value::String(ptr) => {
                    let char_val = state.heap.get_string(ptr, Some(&expression.loc))?.chars().nth(index).unwrap(); 
                    let char_str = char_val.to_string();
                    let new_ptr = state.heap.intern_string(char_str);
                    Ok(Ok(Value::String(new_ptr)))
                },
                Value::Tuple(values) => {
                    Ok(Ok(values[index].clone()))
                },
                Value::List(ptr) => {
                    Ok(Ok(state.heap.get_list(ptr, Some(&expression.loc))?[index].clone()))
                },
                _ => unreachable!()
            }
        },
        ast::Expr::Slice { ref indexed, ref indexer_start, ref indexer_border, ref indexer_step } => {
            let original_indexed_value = eval_or_return_from_expr!(state, indexed, program);
            let indexed_length: usize = get_indexed_length(state, &original_indexed_value, indexed)?;

            let (reverse, indexer_step) = match indexer_step {
                Some(indexer_step) => {
                    let original_indexer_value = eval_or_return_from_expr!(state, indexer_step, program);
                    let indexer_value = match original_indexer_value.clone() {
                        Value::Int(i) => i,
                        _ => return Err(InterpreterErrorMessage {
                            error: InterpreterError::TypeError { 
                                expected: "int".to_string(), 
                                got: original_indexer_value.get_type_name() 
                            },
                            loc: Some(indexer_step.loc.clone())
                        })
                    };

                    if indexer_value < 0 {
                        (true, (-indexer_value) as usize)
                    } else {
                        (false, indexer_value as usize)
                    }
                },
                _ => (false, 1)
            };

            let (indexer_start, _, remaining_elements) = match reverse {
                false => {
                    let value_indexer_start: usize = match indexer_start {
                        Some(indexer) => {
                            let original_indexer_value = eval_or_return_from_expr!(state, indexer, program);
                            soft_wrap_index(original_indexer_value, indexer.loc.clone(), indexed_length)?
                        },
                        _ => 0
                    };

                    let value_indexer_border: usize = match indexer_border {
                        Some(indexer) => {
                            let original_indexer_value = eval_or_return_from_expr!(state, indexer, program);
                            soft_wrap_index(original_indexer_value, indexer.loc.clone(), indexed_length)?
                        },
                        _ => indexed_length
                    };
                    

                    let remaining_elements = {
                        let d = (value_indexer_border as i64) - (value_indexer_start as i64);
                        if d < 0 {
                            0 as usize
                        } else {
                            d as usize
                        }
                    };
                    
                    (value_indexer_start, value_indexer_border, remaining_elements)
                },
                true => {
                    // reversed
                    let value_indexer_start: usize = match indexer_border {
                        Some(indexer) => {
                            let original_indexer_value = eval_or_return_from_expr!(state, indexer, program);
                            soft_wrap_index(original_indexer_value, indexer.loc.clone(), indexed_length)?
                        },
                        _ => 0
                    };

                    // reversed
                    let value_indexer_border: usize = match indexer_start {
                        Some(indexer) => {
                            let original_indexer_value = eval_or_return_from_expr!(state, indexer, program);
                            soft_wrap_index(original_indexer_value, indexer.loc.clone(), indexed_length)?
                        },
                        _ => indexed_length
                    };
                    

                    let remaining_elements = {
                        let d = (value_indexer_border as i64) - (value_indexer_start as i64);
                        if d < 0 {
                            0 as usize
                        } else {
                            d as usize
                        }
                    };

                    (value_indexer_start, value_indexer_border, remaining_elements)
                }
            };

            match original_indexed_value {
                Value::String(ptr) => {
                    let str = state.heap.get_string(ptr, Some(&indexed.loc))?;
                    if reverse {
                        let str: Vec<char> = str.chars().into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).collect();
                        let str = str.into_iter().rev().collect();
                        let new_ptr = state.heap.intern_string(str);
                        Ok(Ok(Value::String(new_ptr)))
                    } else {                        
                        let str: String = str.chars().into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).collect();
                        let new_ptr = state.heap.intern_string(str);
                        Ok(Ok(Value::String(new_ptr)))
                    }                            
                },
                Value::Tuple(values) => {
                    if reverse {
                        Ok(Ok(Value::Tuple(values.into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).rev().collect())))
                    } else {
                        Ok(Ok(Value::Tuple(values.into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).collect())))
                    }
                },
                Value::List(ptr) => {
                    let l = state.heap.get_list(ptr, Some(&indexed.loc))?;
                    if reverse {
                        let new_list = HeapObject::List(l.clone().into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).rev().collect());
                        Ok(Ok(Value::List(state.heap.alloc(new_list))))
                    } else {
                        let new_list = HeapObject::List(l.clone().into_iter().skip(indexer_start).take(remaining_elements).step_by(indexer_step).collect());
                        Ok(Ok(Value::List(state.heap.alloc(new_list))))
                    }
                },
                _ => unreachable!()
            }
        }
        ast::Expr::FunctionPtr(ref s) => {Ok(Ok(Value::FunctionPtr(s.clone())))},
        ast::Expr::Lambda { ref arguments, ref expr } => {
            let mut captured = HashMap::new();
            let argument_names: Vec<&String> = arguments.iter().map(|x| &x.name).collect();
            for (k, v) in ast::LocExpr::free_variables(expr).iter().map(|v| (v, resolve_variable_from_state(state, v, program))) {
                if argument_names.contains(&k) {
                    continue;
                }
                let v = v?;

                match v {
                    Some(v) => captured.insert(k.clone(), v),
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::VariableNotFound(k.clone()), loc: Some(expression.loc.clone())})
                };
            }

            let ptr = state.heap.alloc(HeapObject::Lambda {arguments: arguments.clone(), expr: expr.clone(), captured: captured});
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

fn wrap_index(original_indexer_value: Value, indexer_loc: ast::Loc, indexed_length: usize) -> Result<usize, InterpreterErrorMessage> {
    let mut indexer_value = match original_indexer_value.clone() {
        Value::Int(i) => i,
        _ => return Err(InterpreterErrorMessage {
            error: InterpreterError::TypeError { 
                expected: "int".to_string(), 
                got: original_indexer_value.get_type_name() 
            },
            loc: Some(indexer_loc)
        })
    };

    if indexer_value < 0 {
        indexer_value = (indexed_length as i64) + indexer_value;
    }
    

    if indexer_value < 0 || indexer_value >= (indexed_length as i64) {
        return Err(InterpreterErrorMessage {
            error: InterpreterError::IndexOutOfBounds,
            loc: Some(indexer_loc)
        });
    }

    return Ok(indexer_value as usize);
}

// no bounds check
fn soft_wrap_index(original_indexer_value: Value, indexer_loc: ast::Loc, indexed_length: usize) -> Result<usize, InterpreterErrorMessage> {
    let mut indexer_value = match original_indexer_value.clone() {
        Value::Int(i) => i,
        _ => return Err(InterpreterErrorMessage {
            error: InterpreterError::TypeError { 
                expected: "int".to_string(), 
                got: original_indexer_value.get_type_name() 
            },
            loc: Some(indexer_loc)
        })
    };

    if indexer_value < 0 {
        indexer_value = (indexed_length as i64) + indexer_value;
    }

    return Ok(indexer_value as usize);
}


fn get_indexed_length(state: &mut State, original_indexed_value: &Value, indexed: &ast::LocExpr) -> Result<usize, InterpreterErrorMessage> {
    match &original_indexed_value {
        Value::String(ptr) => {
            Ok(state.heap.get_string(*ptr, Some(&indexed.loc))?.chars().count())
        },
        Value::Tuple(values) => Ok(values.len()),
        Value::List(ptr) => {
            Ok(state.heap.get_list(*ptr, Some(&indexed.loc))?.len())
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
    _: &ast::Loc,
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
            let value = eval_or_return_from_expr!(state, &arg.expr, program);
            let extra_args: Vec<Value> = match value.clone() {
                Value::Tuple(elements) => elements,
                Value::List(ptr) => {
                    state.heap.get_list(ptr, Some(&arg.loc))?.clone()
                },
                _ => return Err(InterpreterErrorMessage {
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
            let value = eval_or_return_from_expr!(state, &arg.expr, program);
            match value.clone() {
                Value::Dictionary(ptr) => {
                    let index_ref = state.heap.get_dict(ptr, Some(&arg.loc))?;

                    for (key, value) in index_ref.iter() {
                        match key.clone() {
                            Value::String(ptr) => {
                                let s = state.heap.get_string(ptr, Some(&arg.loc))?;

                                if !keyword_values.contains_key(&s.clone()) {
                                    keyword_values.insert(s.clone(), (None, value.clone()));
                                }
                            },
                            _ => return Err(InterpreterErrorMessage {
                                error: InterpreterError::TypeError {
                                    expected: "string key".to_string(),
                                    got: key.get_type_name()
                                },
                                loc: Some(arg.loc.clone())
                            })                               
                        }
                    }
                },
                _ => return Err(InterpreterErrorMessage {
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
        let value = eval_or_return_from_expr!(state, &keyword_arg.expr, program);
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
                        let ptr = state.heap.intern_string(String::from(key));
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

    new_values.into_iter().for_each(|(n,v)| {state.stack.shadow_variable(&n, v);});

    let value = run_statement(state, &function.body, program);

    state.stack.drop_function_context();

    match value {
        Ok(StatementReturn::Return(v)) | Ok(StatementReturn::Eval(v)) => {
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
            Ok(heap.get_string(*ptr, Some(arg_loc))?.chars().count() as i64)
        },
        Value::Tuple(v) => Ok(v.len() as i64),
        Value::List(ptr) => {
            Ok(heap.get_list(*ptr, Some(arg_loc))?.len() as i64)
        },
        Value::Dictionary(ptr) => {
            Ok(heap.get_dict(*ptr, Some(arg_loc))?.len() as i64)
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
            let s = state.heap.get_string(*ptr, None)?;
            let new_ptr = state.heap.intern_string(s.clone());
            Ok(Value::String(new_ptr))
        },

        Value::Tuple(values) => {
            let new_values = values.iter()
                .map(|v| deep_clone_value(state, v))
                .collect::<Result<Vec<Value>, _>>()?;
            Ok(Value::Tuple(new_values))
        },

        Value::List(ptr) => {
            let list = state.heap.get_list(*ptr, None)?.clone();
            let new_list = list.iter()
                .map(|v| deep_clone_value(state, v))
                .collect::<Result<Vec<Value>, _>>()?;
            
            let new_ptr = state.heap.alloc(HeapObject::List(new_list));
            Ok(Value::List(new_ptr))
        },

        Value::Dictionary(ptr) => {
            let dict = state.heap.get_dict(*ptr, None)?.clone();
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
            let lambda_obj = match state.heap.get_lambda(*ptr, None)? {
                (args, expr, captured) => (args.clone(), expr.clone(), captured.clone())
            };
            let new_ptr = state.heap.alloc(HeapObject::Lambda { arguments: lambda_obj.0, expr: Box::new(lambda_obj.1), captured: lambda_obj.2 });
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
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
                        let s = state.heap.get_string(*ptr, None)?;
                        
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };

            let result_value = if values.is_empty() {
                // Case 1: bool() -> False
                Value::Bool(false)
            } else {
                // Case 2: bool(x)
                let value_to_convert = &values[0];

                Value::Bool(value_to_convert.truthy(state, loc)?)
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
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
                        iterable_elements = state.heap.get_list(*ptr, Some(&loc))?.clone();
                    },
                    Value::String(ptr) => {
                        let s = state.heap.get_string(*ptr, Some(&loc))?.clone();
                        iterable_elements = s.chars().map(|c| {
                            let s_ptr = state.heap.intern_string(c.to_string());
                            Value::String(s_ptr)
                        }).collect();
                    },
                    Value::Dictionary(ptr) => {
                        iterable_elements = state.heap.get_dict(*ptr, Some(&loc))?.keys().cloned().collect();
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
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
                        iterable_elements = state.heap.get_list(*ptr, Some(&loc))?.clone();
                    },
                    Value::String(ptr) => {
                        let s = state.heap.get_string(*ptr, Some(&loc))?.clone();
                        iterable_elements = s.chars().map(|c| {
                            let s_ptr = state.heap.intern_string(c.to_string());
                            Value::String(s_ptr)
                        }).collect();
                    },
                    Value::Dictionary(ptr) => {
                        iterable_elements = state.heap.get_dict(*ptr, Some(&loc))?.keys().cloned().collect();
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
                Value::List(ptr) => state.heap.get_list(*ptr, Some(&loc))?,
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
                    state.heap.get_list_mut(*ptr, Some(&l_loc))?.push(v_val.clone());
                    Ok(Ok(Value::Void))
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
                    match state.heap.get_list_mut(*ptr, Some(&l_loc))?.pop() {
                        Some(value) => Ok(Ok(value)),
                        _ => Err(InterpreterErrorMessage { error: InterpreterError::IndexOutOfBounds, loc: Some(loc.clone()) })
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
                    let keys_vec = state.heap.get_dict(*ptr, Some(&d_loc))?.keys().cloned().collect::<Vec<Value>>();
                    let new_list_ptr = state.heap.alloc(HeapObject::List(keys_vec));
                    Ok(Ok(Value::List(new_list_ptr)))
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
                    let values_vec =  state.heap.get_dict(*ptr, Some(&d_loc))?.values().cloned().collect::<Vec<Value>>();
                    let new_list_ptr = state.heap.alloc(HeapObject::List(values_vec));
                    Ok(Ok(Value::List(new_list_ptr)))
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
                    let items_vec = state.heap.get_dict(*ptr, Some(&d_loc))?.iter()
                        .map(|(k, v)| Value::Tuple(vec![k.clone(), v.clone()]))
                        .collect::<Vec<Value>>();
                    let new_list_ptr = state.heap.alloc(HeapObject::List(items_vec));
                    Ok(Ok(Value::List(new_list_ptr)))
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "dict".to_string(), got: d_val.get_type_name() },
                    loc: Some(d_loc.clone())
                })
            }
        },

        "read" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("f"), arg_type: None, loc: 0..0}],
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

            let f_val = args_map.get("f").unwrap();
            let f_loc = &positional_arguments.get(0).unwrap().loc;

            return match f_val {
                Value::String(ptr) => {
                    match read_file(state.heap.get_string(*ptr, Some(&f_loc))?) {
                        Ok(s) => {
                            let ptr = state.heap.intern_string(s);
                            return Ok(Ok(Value::String(ptr)));
                        }
                        Err(e) => Err(InterpreterErrorMessage {error: InterpreterError::FileError(format!("Something went wrong reading the file: {}", e)), loc: Some(f_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "str".to_string(), got: f_val.get_type_name() },
                    loc: Some(f_loc.clone())
                })
            }
        },

        "range" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![],
                variadic_argument: Some(ast::Argument {name: String::from("args"), arg_type: None, loc: 0..0}),
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
                return_type: None
            };

            let args = preprocess_args(state, &contract, loc, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, program)?;
            let args = match args {
                Ok(v) => v,
                Err(v) => return Ok(Err(v))
            };

            let values = match args.get("args").unwrap() {
                Value::List(ptr) => state.heap.get_list(*ptr, Some(&loc))?,
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected")), loc: Some(loc.clone())})
            };

            let values: Result<Vec<i64>, InterpreterErrorMessage> = values.iter().map(|x| {
                match x {
                    Value::Int(i) => Ok(i.clone()),
                    _ => Err(InterpreterErrorMessage {error: InterpreterError::ArgumentError(String::from("Range expects int arguments")), loc: Some(loc.clone())})
                }
            }).collect();

            let values = values?;

            let (start, end, step) = match values.len() {
                1 => (0, values[0], 1),
                2 => (values[0], values[1], 1),
                3 => (values[0], values[1], values[2]),
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::ArgumentError(String::from("Range expects 1 to 3 arguments")), loc: Some(loc.clone())})
            };

            if step == 0 {
                return Err(InterpreterErrorMessage {error: InterpreterError::ArgumentError(String::from("Step argument cannot be 0")), loc: Some(loc.clone())})
            }

            let mut result = Vec::new();
            let mut i = start;

            if step > 0 {
                while i < end {
                    result.push(i);
                    i += step;
                }
            } else {
                while i > end {
                    result.push(i);
                    i += step;
                }
            }

            let result: Vec<Value> = result.iter().map(|x| Value::Int(*x)).collect();
            return Ok(Ok(Value::List(state.heap.alloc(HeapObject::List(result)))));
        }

        "read_as_list" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![ast::Argument {name: String::from("f"), arg_type: None, loc: 0..0}],
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

            let f_val = args_map.get("f").unwrap();
            let f_loc = &positional_arguments.get(0).unwrap().loc;

            return match f_val {
                Value::String(ptr) => {
                    match read_file(state.heap.get_string(*ptr, Some(f_loc))?) {
                        Ok(s) => {
                            let list = HeapObject::List(s.chars().map(|x| Value::String(state.heap.intern_string(x.to_string()))).collect());
                            let ptr = state.heap.alloc(list);
                            return Ok(Ok(Value::List(ptr)));
                        }
                        Err(e) => Err(InterpreterErrorMessage {error: InterpreterError::FileError(format!("Something went wrong reading the file: {}", e)), loc: Some(f_loc.clone())})
                    }
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "str".to_string(), got: f_val.get_type_name() },
                    loc: Some(f_loc.clone())
                })
            }
        },

        "split" | "String.split" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![
                    ast::Argument {name: String::from("s"), arg_type: None, loc: 0..0},
                    ast::Argument {name: String::from("sep"), arg_type: None, loc: 0..0},
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

            let s_val = args_map.get("s").unwrap();
            let s_loc = &positional_arguments.get(0).unwrap().loc;

            let sep_val = args_map.get("sep").unwrap();
            let sep_loc = &positional_arguments.get(1).unwrap().loc;
            

            return match (s_val, sep_val) {
                (Value::String(ptr), Value::String(ptr2)) => {
                    let (s, sep) = match (state.heap.get_string(*ptr, Some(&s_loc))?, state.heap.get_string(*ptr2, Some(&sep_loc))?) {
                        (s, sep) => (s.clone(), sep.clone())
                    };

                    let sep_list: Vec<Value> = s.split(&sep).map(|str| {let ptr= state.heap.intern_string(str.to_string()); Value::String(ptr)}).collect();
                    let list_ptr = state.heap.alloc(HeapObject::List(sep_list));
                    Ok(Ok(Value::List(list_ptr)))
                },
                _ => Err(InterpreterErrorMessage {
                    error: InterpreterError::TypeError { expected: "str".to_string(), got: s_val.get_type_name() },
                    loc: Some(s_loc.clone())
                })
            }
        },

        "assert" => {
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
                Value::List(ptr) => state.heap.get_list(*ptr, None)?,
                _ => return Err(InterpreterErrorMessage {error: InterpreterError::InternalError(String::from("List expected from preprocess_args")), loc: Some(loc.clone())})
            };

            if values.is_empty() {
                return Err(InterpreterErrorMessage {error: InterpreterError::AssertionError(String::from("Assertion failed")), loc: Some(loc.clone())})
            } else if values.len() == 1 {
                match values[0].truthy(state, loc)? {
                    true => return Ok(Ok(Value::Void)),
                    false => return Err(InterpreterErrorMessage {error: InterpreterError::AssertionError(String::from("Assertion failed")), loc: Some(loc.clone())})
                }
            } else {
                match values[0].truthy(state, loc)? {
                    true => return Ok(Ok(Value::Void)),
                    false => return Err(InterpreterErrorMessage {error: InterpreterError::AssertionError(format!("Assertion failed {}", DisplayValue::new(&values[1], &state.heap))), loc: Some(loc.clone())})
                }
            }
        },

        "dealloc" => {
            let contract = ast::FunctionPrototype {
                positional_arguments: vec![
                    ast::Argument {name: String::from("object"), arg_type: None, loc: 0..0},
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

            let val = args_map.get("object").unwrap();
            let _loc = &positional_arguments.get(0).unwrap().loc;
            

            match val {
                Value::String(ptr) | Value::List(ptr) | Value::Dictionary(ptr) | Value::Lambda(ptr) => {
                    state.heap.free(*ptr);
                    return Ok(Ok(Value::Void));
                },
                _ => ()
            }
        }

        _ => ()
    }



    return Err(InterpreterErrorMessage {
                error: InterpreterError::FunctionNotFound(function_name.to_string()),
                loc: Some(loc.clone())
            })
}

fn read_file(file_path: &str) -> io::Result<String> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
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

fn run_statement(state: &mut State, stmt: &ast::LocStmt, program: &ast::Program) -> Result<StatementReturn, InterpreterErrorMessage> {
    match &stmt.stmt {
        ast::Stmt::Assignment { target, expr: expression } => {
            match &target.expr {
                ast::Expr::Variable(v) => {
                    let value = eval_or_return_from_stmt!(state, expression, program);
                    state.stack.update_variable(&v,value);
                    return Ok(StatementReturn::None)
                },
                ast::Expr::Indexing { indexed, indexer } => {
                    let original_indexed_value = eval_or_return_from_stmt!(state, indexed, program);
                    let original_indexer_value = eval_or_return_from_stmt!(state, indexer, program);

                    let value = eval_or_return_from_stmt!(state, expression, program);

                    if let Value::Dictionary(ptr) = original_indexed_value {
                        if !original_indexer_value.hashable() {
                             return Err(InterpreterErrorMessage {
                                error: InterpreterError::UnhashableKey,
                                loc: Some(indexer.loc.clone())
                            })
                        }
                        state.heap.get_dict_mut(ptr, Some(&indexed.loc))?.insert(original_indexer_value, value);

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
                            state.heap.get_list_mut(ptr, None)?[index] = value;
                        },
                        _ => unreachable!()
                    }
                },
                ast::Expr::Tuple(elements) | ast::Expr::List(elements) => {
                    let value = eval_or_return_from_stmt!(state, expression, program);
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
        ast::Stmt::FunctionCall { expr: expression } => {let _ = eval_expression(state, expression, program)?;},
        ast::Stmt::Return { expr: expression } => {
            let value = eval_or_return_from_stmt!(state, expression, program);
            return Ok(StatementReturn::Return(value));
        },
        ast::Stmt::IfElse { cond: condition, if_body, else_body } => {
            let eval_condition = eval_or_return_from_stmt!(state, &condition, program);

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
            let eval_condition = eval_or_return_from_stmt!(state, condition, program);

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

                let eval_condition = eval_or_return_from_stmt!(state, condition, program);
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
            let value = eval_or_return_from_stmt!(state, expression, program);
            return Ok(StatementReturn::Eval(value));
        }
    }

    return Ok(StatementReturn::None);
}


fn unpack_elements(state: &State, variables: &Vec<ast::LocExpr>, value: Value, value_loc: &ast::Loc) -> Result<Vec<(String, Value)>, InterpreterErrorMessage> {
    let values = match value.clone() {
        Value::Tuple(elements) => elements,
        Value::List(ptr) => {
            state.heap.get_list(ptr, Some(&value_loc))?.clone()
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