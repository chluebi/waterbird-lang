use std::ops::Range;

type Loc = Range<usize>;

enum Type {
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

enum TypeLiteral {
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

struct LocTypeLiteral {
    expr: TypeLiteral,
    loc: Loc
}

enum BinOp {}

enum UnOp {}

pub struct CallArgument {
    pub expression: Box<LocExpr>,
    pub loc: Loc 
}


pub struct CallKeywordArgument {
    pub name: String,
    pub expression: Box<LocExpr>,
    pub loc: Loc 
}


enum Expr {
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
        function_name: String,
        positional_arguments: Vec<CallArgument>,
        variadic_argument: Option<CallArgument>,
        keyword_arguments: Vec<CallKeywordArgument>,
        keyword_variadic_argument: Option<CallArgument>
    },
    Indexing {
        indexed: Box<LocExpr>,
        indexer: Box<LocExpr>
    }
}

struct LocExpr {
    expr: Expr,
    loc: Loc
}
