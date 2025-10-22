use std::ops::Range;
use std::collections::HashMap;

pub type Loc = Range<usize>;

pub enum Type {
    Generic(String),
    Int32,
    Int64,
    Bool,
    Str,
    Tuple(Vec<Type>),
    List(Box<Type>),
    Dict {
        keys: Box<Type>,
        values: Box<Type>
    }
}

pub enum TypeLiteral {
    Generic(String),
    Int32,
    Int64,
    Bool,
    Str,
    Tuple(Vec<LocTypeLiteral>),
    List(Box<LocTypeLiteral>),
    Dict {
        keys: Box<LocTypeLiteral>,
        values: Box<LocTypeLiteral>
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
    pub keyword_variadic_argument: Option<Argument>
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
