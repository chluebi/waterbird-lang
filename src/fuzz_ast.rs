use serde::{Deserialize, Serialize};
use autarkie;
use std::collections::HashMap;
use crate::ast;

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentType {
    pub name: String,
    pub arg_type: Type
}

impl KeywordArgumentType {
    pub fn to_ast(&self) -> ast::KeywordArgumentType {
        ast::KeywordArgumentType {
            name: self.name.clone(),
            arg_type: self.arg_type.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Unknown, // Top
    Impossible, // Bottom
    Unit,
    Generic(String),
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
        variadic_argument: Option<Box<Type>>,
        keyword_arguments: Vec<KeywordArgumentType>,
        keyword_variadic_argument: Option<Box<Type>>,
        return_type: Box<Type>
    }
}

impl Type {
    pub fn to_ast(&self) -> ast::Type {
        match self {
            Type::Unknown => ast::Type::Unknown,
            Type::Impossible => ast::Type::Impossible,
            Type::Unit => ast::Type::Unit,
            Type::Generic(s) => ast::Type::Generic(s.clone()),
            Type::Int => ast::Type::Int,
            Type::Bool => ast::Type::Bool,
            Type::Str => ast::Type::Str,
            Type::Tuple(v) => ast::Type::Tuple(v.iter().map(|t| t.to_ast()).collect()),
            Type::List(t) => ast::Type::List(Box::new(t.to_ast())),
            Type::Dict { keys, values } => ast::Type::Dict {
                keys: Box::new(keys.to_ast()),
                values: Box::new(values.to_ast()),
            },
            Type::Callable {
                generics,
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type,
            } => ast::Type::Callable {
                generics: generics.clone(),
                positional_arguments: positional_arguments.iter().map(|t| t.to_ast()).collect(),
                variadic_argument: variadic_argument.as_ref().map(|t| Box::new(t.to_ast())),
                keyword_arguments: keyword_arguments.iter().map(|k| k.to_ast()).collect(),
                keyword_variadic_argument: keyword_variadic_argument.as_ref().map(|t| Box::new(t.to_ast())),
                return_type: Box::new(return_type.to_ast()),
            },
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub struct GenericLiteral {
    pub name: String,
}

impl GenericLiteral {
    pub fn to_ast(&self) -> ast::GenericLiteral {
        ast::GenericLiteral {
            name: self.name.clone(),
            loc: 0..0,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentTypeLiteral {
    pub name: String,
    pub arg_type: Box<LocTypeLiteral>
}

impl KeywordArgumentTypeLiteral {
    pub fn to_ast(&self) -> ast::KeywordArgumentTypeLiteral {
        ast::KeywordArgumentTypeLiteral {
            name: self.name.clone(),
            arg_type: Box::new(self.arg_type.to_ast()),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
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

impl TypeLiteral {
    pub fn to_ast(&self) -> ast::TypeLiteral {
        match self {
            TypeLiteral::Generic(s) => ast::TypeLiteral::Generic(s.clone()),
            TypeLiteral::Void => ast::TypeLiteral::Void,
            TypeLiteral::Int => ast::TypeLiteral::Int,
            TypeLiteral::Bool => ast::TypeLiteral::Bool,
            TypeLiteral::Str => ast::TypeLiteral::Str,
            TypeLiteral::Tuple(v) => ast::TypeLiteral::Tuple(v.iter().map(|t| t.to_ast()).collect()),
            TypeLiteral::List(t) => ast::TypeLiteral::List(Box::new(t.to_ast())),
            TypeLiteral::Dict { keys, values } => ast::TypeLiteral::Dict {
                keys: Box::new(keys.to_ast()),
                values: Box::new(values.to_ast()),
            },
            TypeLiteral::Callable {
                generics,
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type,
            } => ast::TypeLiteral::Callable {
                generics: generics.clone(),
                positional_arguments: positional_arguments.iter().map(|t| t.to_ast()).collect(),
                variadic_argument: Box::new(variadic_argument.as_ref().clone().map(|t| t.to_ast())),
                keyword_arguments: keyword_arguments.iter().map(|k| k.to_ast()).collect(),
                keyword_variadic_argument: Box::new(keyword_variadic_argument.as_ref().clone().map(|t| t.to_ast())),
                return_type: Box::new(return_type.to_ast()),
            },
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub struct LocTypeLiteral {
    pub typ: TypeLiteral,
}

impl LocTypeLiteral {
    pub fn to_ast(&self) -> ast::LocTypeLiteral {
        ast::LocTypeLiteral {
            typ: self.typ.to_ast(),
            loc: 0..0,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    Eq, Neq, Leq, Geq, Lt, Gt,
    Add, Sub, Mul, Div, Mod,
    ShiftLeft, ShiftRightArith,
    And, Or, In
}

impl BinOp {
    pub fn to_ast(&self) -> ast::BinOp {
        match self {
            BinOp::Eq => ast::BinOp::Eq,
            BinOp::Neq => ast::BinOp::Neq,
            BinOp::Leq => ast::BinOp::Leq,
            BinOp::Geq => ast::BinOp::Geq,
            BinOp::Lt => ast::BinOp::Lt,
            BinOp::Gt => ast::BinOp::Gt,
            BinOp::Add => ast::BinOp::Add,
            BinOp::Sub => ast::BinOp::Sub,
            BinOp::Mul => ast::BinOp::Mul,
            BinOp::Div => ast::BinOp::Div,
            BinOp::Mod => ast::BinOp::Mod,
            BinOp::ShiftLeft => ast::BinOp::ShiftLeft,
            BinOp::ShiftRightArith => ast::BinOp::ShiftRightArith,
            BinOp::And => ast::BinOp::And,
            BinOp::Or => ast::BinOp::Or,
            BinOp::In => ast::BinOp::In,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not
}

impl UnOp {
    pub fn to_ast(&self) -> ast::UnOp {
        match self {
            UnOp::Neg => ast::UnOp::Neg,
            UnOp::Not => ast::UnOp::Not,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct CallArgument {
    pub expr: Box<LocExpr>,
}

impl CallArgument {
    pub fn to_ast(&self) -> ast::CallArgument {
        ast::CallArgument {
            expr: Box::new(self.expr.to_ast()),
            loc: 0..0,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct CallKeywordArgument {
    pub name: String,
    pub expr: Box<LocExpr>,
}

impl CallKeywordArgument {
    pub fn to_ast(&self) -> ast::CallKeywordArgument {
        ast::CallKeywordArgument {
            name: self.name.clone(),
            expr: Box::new(self.expr.to_ast()),
            loc: 0..0,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct LambdaArgument {
    pub name: String,
    pub arg_type_literal: Option<LocTypeLiteral>,
    pub typ: Type
}

impl LambdaArgument {
    pub fn to_ast(&self) -> ast::LambdaArgument {
        ast::LambdaArgument {
            name: self.name.clone(),
            arg_type_literal: self.arg_type_literal.as_ref().map(LocTypeLiteral::to_ast),
            loc: 0..0,
            typ: self.typ.to_ast()
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
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
    Slice {
        indexed: Box<LocExpr>,
        indexer_start: Option<Box<LocExpr>>,
        indexer_border: Option<Box<LocExpr>>,
        indexer_step: Option<Box<LocExpr>>,
    },
    Lambda {
       arguments: Vec<LambdaArgument>,
       expr: Box<LocExpr>
    },
    Block {
        statements: Vec<LocStmt>    
    }
}

impl Expr {
    pub fn to_ast(&self) -> ast::Expr {
        match self {
            Expr::Variable(s) => ast::Expr::Variable(s.clone()),
            Expr::DotAccess(e, s) => ast::Expr::DotAccess(Box::new(e.to_ast()), s.clone()),
            Expr::Int(i) => ast::Expr::Int(*i),
            Expr::Bool(b) => ast::Expr::Bool(*b),
            Expr::Str(s) => ast::Expr::Str(s.clone()),
            Expr::Tuple(v) => ast::Expr::Tuple(v.iter().map(|e| e.to_ast()).collect()),
            Expr::List(v) => ast::Expr::List(v.iter().map(|e| e.to_ast()).collect()),
            Expr::Dictionary(v) => ast::Expr::Dictionary(v.iter().map(|(k, val)| (k.to_ast(), val.to_ast())).collect()),
            Expr::BinOp { op, left, right } => ast::Expr::BinOp {
                op: op.to_ast(),
                left: Box::new(left.to_ast()),
                right: Box::new(right.to_ast()),
            },
            Expr::UnOp { op, expr } => ast::Expr::UnOp {
                op: op.to_ast(),
                expr: Box::new(expr.to_ast()),
            },
            Expr::FunctionCall {
                function,
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
            } => ast::Expr::FunctionCall {
                function: Box::new(function.to_ast()),
                positional_arguments: positional_arguments.iter().map(|a| a.to_ast()).collect(),
                variadic_argument: variadic_argument.as_ref().map(|a| a.to_ast()),
                keyword_arguments: keyword_arguments.iter().map(|a| a.to_ast()).collect(),
                keyword_variadic_argument: keyword_variadic_argument.as_ref().map(|a| a.to_ast()),
            },
            Expr::Indexing { indexed, indexer } => ast::Expr::Indexing {
                indexed: Box::new(indexed.to_ast()),
                indexer: Box::new(indexer.to_ast()),
            },
            Expr::Slice { indexed, indexer_start, indexer_border, indexer_step } => ast::Expr::Slice {
                indexed: Box::new(indexed.to_ast()),
                indexer_start: indexer_start.as_ref().map(|e| Box::new(e.to_ast())),
                indexer_border: indexer_border.as_ref().map(|e| Box::new(e.to_ast())),
                indexer_step: indexer_step.as_ref().map(|e| Box::new(e.to_ast())),
            },
            Expr::Lambda { arguments, expr } => ast::Expr::Lambda {
                arguments: arguments.iter().map(|a| a.to_ast()).collect(),
                expr: Box::new(expr.to_ast()),
            },
            Expr::Block { statements } => ast::Expr::Block {
                statements: statements.iter().map(|s| s.to_ast()).collect(),
            },
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct LocExpr {
    pub expr: Expr,
    pub typ: Type
}

impl LocExpr {
    pub fn to_ast(&self) -> ast::LocExpr {
        ast::LocExpr {
            expr: self.expr.to_ast(),
            loc: 0..0,
            typ: self.typ.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
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
    SoftBlock { 
        statements: Vec<LocStmt>
    },
    Expression {
        expr: LocExpr
    },
    Break,
    Continue
}

impl Stmt {
    pub fn to_ast(&self) -> ast::Stmt {
        match self {
            Stmt::Assignment { target, expr } => ast::Stmt::Assignment {
                target: target.to_ast(),
                expr: expr.to_ast(),
            },
            Stmt::FunctionCall { expr } => ast::Stmt::FunctionCall {
                expr: expr.to_ast(),
            },
            Stmt::Return { expr } => ast::Stmt::Return {
                expr: expr.to_ast(),
            },
            Stmt::IfElse { cond, if_body, else_body } => ast::Stmt::IfElse {
                cond: cond.to_ast(),
                if_body: Box::new(if_body.to_ast()),
                else_body: Box::new(else_body.to_ast()),
            },
            Stmt::While { cond, body } => ast::Stmt::While {
                cond: cond.to_ast(),
                body: Box::new(body.to_ast()),
            },
            Stmt::Block { statements } => ast::Stmt::Block {
                statements: statements.iter().map(|s| s.to_ast()).collect(),
            },
            Stmt::SoftBlock { statements } => ast::Stmt::SoftBlock {
                statements: statements.iter().map(|s| s.to_ast()).collect(),
            },
            Stmt::Expression { expr } => ast::Stmt::Expression {
                expr: expr.to_ast(),
            },
            Stmt::Break => ast::Stmt::Break,
            Stmt::Continue => ast::Stmt::Continue,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct LocStmt {
    pub stmt: Stmt,
    pub typ: Type 
}

impl LocStmt {
    pub fn to_ast(&self) -> ast::LocStmt {
        ast::LocStmt {
            stmt: self.stmt.to_ast(),
            loc: 0..0,
            typ: self.typ.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub arg_type_literal: Option<LocTypeLiteral>,
    pub typ: Type,
}

impl Argument {
    pub fn to_ast(&self) -> ast::Argument {
        ast::Argument {
            name: self.name.clone(),
            arg_type_literal: self.arg_type_literal.as_ref().map(|t| t.to_ast()),
            loc: 0..0,
            typ: self.typ.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct KeywordArgument {
    pub name: String,
    pub expr: LocExpr,
    pub arg_type_literal: Option<LocTypeLiteral>,
    pub typ: Type
}

impl KeywordArgument {
    pub fn to_ast(&self) -> ast::KeywordArgument {
        ast::KeywordArgument {
            name: self.name.clone(),
            expr: self.expr.to_ast(),
            arg_type_literal: self.arg_type_literal.as_ref().map(|t| t.to_ast()),
            loc: 0..0,
            typ: self.typ.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct FunctionPrototype {
    pub generics: Vec<GenericLiteral>,
    pub positional_arguments: Vec<Argument>,
    pub variadic_argument: Option<Argument>,
    pub keyword_arguments: Vec<KeywordArgument>,
    pub keyword_variadic_argument: Option<Argument>,
    pub return_type_literal: Option<LocTypeLiteral>,
    pub return_typ: Type,
    pub typ: Type
}

impl FunctionPrototype {
    pub fn to_ast(&self) -> ast::FunctionPrototype {
        ast::FunctionPrototype {
            generics: self.generics.iter().map(|g| g.to_ast()).collect(),
            positional_arguments: self.positional_arguments.iter().map(|a| a.to_ast()).collect(),
            variadic_argument: self.variadic_argument.as_ref().map(|a| a.to_ast()),
            keyword_arguments: self.keyword_arguments.iter().map(|k| k.to_ast()).collect(),
            keyword_variadic_argument: self.keyword_variadic_argument.as_ref().map(|a| a.to_ast()),
            return_type_literal: self.return_type_literal.as_ref().map(|t| t.to_ast()),
            return_typ: self.return_typ.to_ast(),
            typ: self.typ.to_ast(),
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct Function {
    pub name: String,
    pub contract: FunctionPrototype,
    pub body: Box<LocStmt>,
}

impl Function {
    pub fn to_ast(&self) -> ast::Function {
        ast::Function {
            name: self.name.clone(),
            contract: self.contract.to_ast(),
            body: Box::new(self.body.to_ast()),
            loc: 0..0,
        }
    }
}

#[derive(Serialize, Deserialize, autarkie::Grammar, Debug, Clone)]
pub struct FuzzData {
    pub functions: Vec<Function>
}

impl FuzzData {
    pub fn to_ast(&self) -> ast::Program {
        let mut functions_map = HashMap::new();
        for func in &self.functions {
            functions_map.insert(func.name.clone(), func.to_ast());
        }
        ast::Program {
            functions: functions_map
        }
    }
}

autarkie::fuzz_libfuzzer!(FuzzData);