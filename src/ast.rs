use std::fmt;
use std::ops::Range;
use std::collections::HashMap;

pub type Loc = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentType {
    pub name: String,
    pub arg_type: Box<Type>
}

impl fmt::Display for KeywordArgumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.arg_type)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Unknown,
    Generic(String),
    Void,
    Int,
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
            Type::Unknown => write!(f, "unknown"),
            Type::Generic(s) => write!(f, "{}", s),
            Type::Void => write!(f, "()"),
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "str"),
            Type::Tuple(type_vec) => {
                type_vec.iter().map(|t| write!(f, "{}", t)).collect()
            },
            Type::List(t) => write!(f, "List<{}>", t),
            Type::Dict{keys, values} => write!(f, "Dict<{},{}>", keys, values),
            Type::Callable{
                generics, 
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type
            } => {
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


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentTypeLiteral {
    pub name: String,
    pub arg_type: Box<LocTypeLiteral>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeLiteral {
    Generic(String),
    Void,
    Int,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocTypeLiteral {
    pub expr: TypeLiteral,
    pub loc: Loc
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    Eq,
    Neq,

    Leq,
    Geq,
    Lt,
    Gt,

    Add,
    Sub,
    Mul,
    Div,

    Mod,

    ShiftLeft,
    ShiftRightArith,

    And,
    Or
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not
}

#[derive(Debug, Clone)]
pub struct CallArgument {
    pub expr: Box<LocExpr>,
    pub loc: Loc 
}


#[derive(Debug, Clone)]
pub struct CallKeywordArgument {
    pub name: String,
    pub expr: Box<LocExpr>,
    pub loc: Loc 
}


#[derive(Debug, Clone)]
pub struct LambdaArgument {
    pub name: String,
    pub loc: Loc
}

#[derive(Debug, Clone)]
pub enum Expr {
    Variable(String),
    Int(i64),
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
    UnOp {
        op: UnOp,
        expr: Box<LocExpr>
    },
    FunctionCall {
        function: Box<LocExpr>,
        positional_arguments: Vec<CallArgument>,
        variadic_argument: Option<CallArgument>,
        keyword_arguments: Vec<CallKeywordArgument>,
        keyword_variadic_argument: Option<CallArgument>
    },
    Indexing {
        indexed: Box<LocExpr>,
        indexer: Box<LocExpr>
    },
    FunctionPtr(String),
    Lambda {
       arguments: Vec<LambdaArgument>,
       expr: Box<LocExpr>
    },
    Block {
        statements: Vec<LocStmt>    
    }
}

#[derive(Debug, Clone)]
pub struct LocExpr {
    pub expr: Expr,
    pub loc: Loc,
    pub typ: Type
}


#[derive(Debug, Clone)]
pub enum Stmt {
    Assignment {
        target: LocExpr,
        expr: LocExpr,
    },
    FunctionCall {
        expr: LocExpr
    },
    Return {
        expr: LocExpr
    },
    IfElse {
        cond: LocExpr,
        if_body: Box<LocStmt>,
        else_body: Box<LocStmt> 
    },
    While {
        cond: LocExpr,
        body: Box<LocStmt>
    },
    Block {
        statements: Vec<LocStmt> 
    },
    Expression {
        expr: LocExpr
    }
}


#[derive(Debug, Clone)]
pub struct LocStmt {
    pub stmt: Stmt,
    pub loc: Loc 
}

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub arg_type: Option<LocTypeLiteral>,
    pub loc: Loc 
}

#[derive(Debug, Clone)]
pub struct KeywordArgument {
    pub name: String,
    pub expr: LocExpr,
    pub loc: Loc
}


#[derive(Debug, Clone)]
pub struct FunctionPrototype {
    pub positional_arguments: Vec<Argument>,
    pub variadic_argument: Option<Argument>,
    pub keyword_arguments: Vec<KeywordArgument>,
    pub keyword_variadic_argument: Option<Argument>,
    pub return_type: Option<LocTypeLiteral>
}


#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub contract: FunctionPrototype,
    pub body: Box<LocStmt>,
    pub loc: Loc 
}

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: HashMap<String, Function>
}
