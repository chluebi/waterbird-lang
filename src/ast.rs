use std::fmt;
use std::ops::Range;
use std::collections::HashMap;

pub type Loc = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentType {
    name: String,
    arg_type: Box<Type>
}

impl fmt::Display for KeywordArgumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.arg_type)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Generic(String),
    Void,
    Int32,
    Int64,
    Bool,
    Str,
    Tuple(Vec<Type>),
    List(Box<Type>),
    Dict {
        keys: Box<Type>,
        values: Box<Type>
    },
    Callable {
        generics: Vec<String>,
        positional_arguments: Vec<Type>,
        variadic_argument: Box<Option<Type>>,
        keyword_arguments: Vec<KeywordArgumentType>,
        keyword_variadic_argument: Box<Option<Type>>,
        return_type: Box<Type>
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Generic(s) => write!(f, "{}", s),
            Type::Void => write!(f, "()"),
            Type::Int32 => write!(f, "int"),
            Type::Int64 => write!(f, "int64"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "str"),
            Type::Tuple(type_vec) => {
                type_vec.iter().map(|t| write!(f, "{}", t)).collect()
            },
            Type::List(t) => write!(f, "List<{}>", t),
            Type::Dict{keys, values} => write!(f, "Dict<{},{}>", keys, values),
            Type::Callable{generics, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, return_type} => {
                write!(f, "fn")?;

                // Generics, if any
                if !generics.is_empty() {
                    let gens = generics.join(", ");
                    write!(f, "(gen: {}; ", gens)?;
                } else {
                    write!(f, "(")?;
                }

                // Positional args
                let mut parts = vec![];

                for t in positional_arguments {
                    parts.push(format!("{}", t));
                }

                // *args
                if let Some(var) = variadic_argument.as_ref() {
                    parts.push(format!("*{}", var));
                }

                // keyword args
                for kw in keyword_arguments {
                    parts.push(format!("{}", kw));
                }

                // **kwargs
                if let Some(kv) = keyword_variadic_argument.as_ref() {
                    parts.push(format!("**{}", kv));
                }

                write!(f, "{}", parts.join(", "))?;

                // Close fn args
                write!(f, ") -> {}", return_type)
            }         
        }
    }
}

pub struct KeywordArgumentTypeLiteral {
    name: String,
    arg_type: Box<LocTypeLiteral>
}

pub enum TypeLiteral {
    Generic(String),
    Void,
    Int32,
    Int64,
    Bool,
    Str,
    Tuple(Vec<LocTypeLiteral>),
    List(Box<LocTypeLiteral>),
    Dict {
        keys: Box<LocTypeLiteral>,
        values: Box<LocTypeLiteral>
    },
    Callable {
        generics: Vec<String>,
        positional_arguments: Vec<LocTypeLiteral>,
        variadic_argument: Box<Option<LocTypeLiteral>>,
        keyword_arguments: Vec<KeywordArgumentTypeLiteral>,
        keyword_variadic_argument: Box<Option<LocTypeLiteral>>,
        return_type: Box<LocTypeLiteral>
    }
}

pub struct LocTypeLiteral {
    expr: TypeLiteral,
    loc: Loc
}

pub enum BinOp {}

pub enum UnOp {}

pub struct CallArgument {
    pub expression: Box<LocExpr>,
    pub loc: Loc 
}


pub struct CallKeywordArgument {
    pub name: String,
    pub expression: Box<LocExpr>,
    pub loc: Loc 
}


pub struct LambdaArgument {
    pub name: String,
    pub loc: Loc
}

pub enum Expr {
    Variable(String),
    Int32(i32),
    Int64(i64),
    Bool(bool),
    Str(String),
    Tuple(Vec<LocExpr>),
    List(Vec<LocExpr>),
    Dictionary(Vec<(LocExpr,LocExpr)>),
    BinOp {
        op: BinOp,
        left: Box<LocExpr>,
        right: Box<LocExpr>,
    },
    Unop {
        op: BinOp,
        expr: Box<LocExpr>
    },
    FunctionCall {
        function_name: Box<LocExpr>,
        positional_arguments: Vec<CallArgument>,
        variadic_argument: Option<CallArgument>,
        keyword_arguments: Vec<CallKeywordArgument>,
        keyword_variadic_argument: Option<CallArgument>
    },
    Indexing {
        indexed: Box<LocExpr>,
        indexer: Box<LocExpr>
    },
    Lambda {
       arguments: Vec<LambdaArgument>,
       expr: Box<LocExpr>
    }
}

pub struct LocExpr {
    expr: Expr,
    loc: Loc
}


pub enum Stmt {
    Assignment {
        target: LocExpr,
        expression: LocExpr,
    },
    FunctionCall {
        expression: LocExpr
    },
    ListAppend {
        target: LocExpr,
        value: LocExpr
    },
    Return {
        expression: LocExpr
    },
    IfElse {
        condition: LocExpr,
        if_body: Body,
        else_body: Body
    },
    While {
        condition: LocExpr,
        body: Body
    }
}


pub struct LocStmt {
    pub statement: Stmt,
    pub loc: Loc 
}


pub struct Body {
    pub statements: Vec<LocStmt>
}

pub struct Argument {
    pub name: String,
    pub arg_type: LocTypeLiteral,
    pub loc: Loc 
}


pub struct KeywordArgument {
    pub name: String,
    pub expression: LocExpr,
    pub loc: Loc
}


pub struct FunctionPrototype {
    pub positional_arguments: Vec<Argument>,
    pub variadic_argument: Option<Argument>,
    pub keyword_arguments: Vec<KeywordArgument>,
    pub keyword_variadic_argument: Option<Argument>,
    pub return_type: LocTypeLiteral
}


pub struct Function {
    pub name: String,
    pub contract: FunctionPrototype,
    pub body: Body,
    pub loc: Loc 
}

pub struct Program {
    pub functions: HashMap<String, Function>
}
