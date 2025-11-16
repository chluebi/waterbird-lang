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
                let parts: Vec<String> = type_vec.iter().map(|t| format!("{}", t)).collect();
                write!(f, "({})", parts.join(", "))
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

impl fmt::Display for KeywordArgumentTypeLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.arg_type)
    }
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

impl fmt::Display for TypeLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeLiteral::Generic(s) => write!(f, "{}", s),
            TypeLiteral::Void => write!(f, "()"),
            TypeLiteral::Int => write!(f, "int"),
            TypeLiteral::Bool => write!(f, "bool"),
            TypeLiteral::Str => write!(f, "str"),
            TypeLiteral::Tuple(type_vec) => {
                let parts: Vec<String> = type_vec.iter().map(|t| format!("{}", t)).collect();
                write!(f, "({})", parts.join(", "))
            },
            TypeLiteral::List(t) => write!(f, "List<{}>", t),
            TypeLiteral::Dict{ keys, values } => write!(f, "Dict<{},{}>", keys, values),
            TypeLiteral::Callable {
                generics,
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type
            } => {
                write!(f, "fn")?;
                if !generics.is_empty() {
                    write!(f, "(gen: {}; ", generics.join(", "))?;
                } else {
                    write!(f, "(")?;
                }
                let mut parts = vec![];
                for t in positional_arguments { parts.push(format!("{}", t)); }
                if let Some(var) = variadic_argument.as_ref() { parts.push(format!("*{}", var)); }
                for kw in keyword_arguments { parts.push(format!("{}", kw)); }
                if let Some(kv) = keyword_variadic_argument.as_ref() { parts.push(format!("**{}", kv)); }
                write!(f, "{}", parts.join(", "))?;
                write!(f, ") -> {}", return_type)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocTypeLiteral {
    pub expr: TypeLiteral,
    pub loc: Loc
}

impl fmt::Display for LocTypeLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
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
    Or,

    In
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self {
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Leq => "<=",
            BinOp::Geq => ">=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::ShiftLeft => "<<",
            BinOp::ShiftRightArith => ">>",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::In => "in"
        };
        write!(f, "{}", op_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_str = match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
        };
        write!(f, "{}", op_str)
    }
}

#[derive(Debug, Clone)]
pub struct CallArgument {
    pub expr: Box<LocExpr>,
    pub loc: Loc 
}

impl fmt::Display for CallArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[derive(Debug, Clone)]
pub struct CallKeywordArgument {
    pub name: String,
    pub expr: Box<LocExpr>,
    pub loc: Loc 
}

impl fmt::Display for CallKeywordArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.expr)
    }
}

#[derive(Debug, Clone)]
pub struct LambdaArgument {
    pub name: String,
    pub loc: Loc
}

impl fmt::Display for LambdaArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Variable(String),
    DotAccess(Box<LocExpr>, String),
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::DotAccess(expr, member) => write!(f, "{}.{}", expr, member),
            Expr::Int(i) => write!(f, "{}", i),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Str(s) => write!(f, "\"{}\"", s.replace("\"", "\\\"")), // Simple escaping
            Expr::Tuple(exprs) => {
                let parts: Vec<String> = exprs.iter().map(|e| format!("{}", e)).collect();
                // Add trailing comma for single-element tuple
                if exprs.len() == 1 {
                    write!(f, "({},)", parts[0])
                } else {
                    write!(f, "({})", parts.join(", "))
                }
            },
            Expr::List(exprs) => {
                let parts: Vec<String> = exprs.iter().map(|e| format!("{}", e)).collect();
                write!(f, "[{}]", parts.join(", "))
            },
            Expr::Dictionary(pairs) => {
                let parts: Vec<String> = pairs.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                write!(f, "{{{}}}", parts.join(", "))
            },
            Expr::BinOp { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            },
            Expr::UnOp { op, expr } => {
                write!(f, "({}{})", op, expr)
            },
            Expr::FunctionCall { function, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument } => {
                write!(f, "{}(", function)?;
                let mut parts = vec![];
                for arg in positional_arguments { parts.push(format!("{}", arg)); }
                if let Some(var) = variadic_argument { parts.push(format!("*{}", var)); }
                for kw in keyword_arguments { parts.push(format!("{}", kw)); }
                if let Some(kv) = keyword_variadic_argument { parts.push(format!("**{}", kv)); }
                write!(f, "{})", parts.join(", "))
            },
            Expr::Indexing { indexed, indexer } => {
                write!(f, "{}[{}]", indexed, indexer)
            },
            Expr::FunctionPtr(name) => write!(f, "<fn_ptr: {}>", name),
            Expr::Lambda { arguments, expr } => {
                let args: Vec<String> = arguments.iter().map(|a| format!("{}", a)).collect();
                write!(f, "lambda({}): {}", args.join(", "), expr)
            },
            Expr::Block { statements } => {
                writeln!(f, "#{{")?;
                for stmt in statements {
                    let stmt_str = format!("{}", stmt);
                    for line in stmt_str.lines() {
                        writeln!(f, "    {}", line)?;
                    }
                    writeln!(f, ";")?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocExpr {
    pub expr: Expr,
    pub loc: Loc,
    pub typ: Type
}

impl fmt::Display for LocExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
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

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::Assignment { target, expr } => {
                write!(f, "{} = {}", target, expr)
            },
            Stmt::FunctionCall { expr } => {
                write!(f, "{}", expr)
            },
            Stmt::Return { expr } => {
                write!(f, "return {}", expr)
            },
            Stmt::IfElse { cond, if_body, else_body } => {
                write!(f, "if {} {}", cond, if_body)?;
                
                if let Stmt::Block { statements } = &else_body.stmt {
                    if statements.is_empty() {
                        return Ok(());
                    }
                }

                write!(f, " else {}", else_body)
            },
            Stmt::While { cond, body } => {
                write!(f, "while {} {}", cond, body)
            },
            Stmt::Block { statements } => {
                writeln!(f, "{{")?;
                for stmt in statements {
                    let stmt_str = format!("{}", stmt);
                    for line in stmt_str.lines() {
                        writeln!(f, "    {}", line)?;
                    }
                }
                write!(f, "}}")
            },
            Stmt::Expression { expr } => {
                write!(f, "{}", expr)
            }
        }
    }
}


#[derive(Debug, Clone)]
pub struct LocStmt {
    pub stmt: Stmt,
    pub loc: Loc 
}

impl fmt::Display for LocStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.stmt)
    }
}

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub arg_type: Option<LocTypeLiteral>,
    pub loc: Loc 
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(typ) = &self.arg_type {
            write!(f, ": {}", typ)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KeywordArgument {
    pub name: String,
    pub expr: LocExpr,
    pub loc: Loc
}

impl fmt::Display for KeywordArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.expr)
    }
}


#[derive(Debug, Clone)]
pub struct FunctionPrototype {
    pub positional_arguments: Vec<Argument>,
    pub variadic_argument: Option<Argument>,
    pub keyword_arguments: Vec<KeywordArgument>,
    pub keyword_variadic_argument: Option<Argument>,
    pub return_type: Option<LocTypeLiteral>
}

impl fmt::Display for FunctionPrototype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        let mut parts = vec![];
        for arg in &self.positional_arguments { parts.push(format!("{}", arg)); }
        if let Some(var) = &self.variadic_argument { parts.push(format!("*{}", var.name)); }
        for kw in &self.keyword_arguments { parts.push(format!("{}", kw)); }
        if let Some(kv) = &self.keyword_variadic_argument { parts.push(format!("**{}", kv.name)); }
        
        write!(f, "{}", parts.join(", "))?;
        write!(f, ")")?;

        if let Some(ret) = &self.return_type {
            write!(f, " -> {}", ret)?;
        }
        Ok(())
    }
}


#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub contract: FunctionPrototype,
    pub body: Box<LocStmt>,
    pub loc: Loc 
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}{} {}", self.name, self.contract, self.body)
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: HashMap<String, Function>
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut functions: Vec<_> = self.functions.values().collect();
        // Sort by name for deterministic output
        functions.sort_by(|a, b| a.name.cmp(&b.name));
        
        for (i, func) in functions.iter().enumerate() {
            if i > 0 {
                // Add two newlines between functions for separation
                writeln!(f, "\n")?;
            }
            write!(f, "{}", func)?;
        }
        Ok(())
    }
}