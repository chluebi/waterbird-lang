use std::fmt;
use std::ops::Range;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast::{self, Loc};

// unique variables
static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn get_unique_var() -> String {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    
    // # is not allowed in variable names by the parser
    format!("#temp_var_{}", id)
}


#[derive(Debug, Clone)]
pub enum PreprocessingError {
    FunctionProcessingError(String)
}

impl fmt::Display for PreprocessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionProcessingError(s) => write!(f, "Error Processing Function: {}", s)
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessingErrorMessage {
    pub error: PreprocessingError,
    pub loc: Option<Loc>
}

impl fmt::Display for PreprocessingErrorMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}




#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordArgumentTypeLiteral {
    name: String,
    arg_type: Box<LocTypeLiteral>
}

impl KeywordArgumentTypeLiteral {
    fn preprocess(t: Self) -> Result<ast::KeywordArgumentTypeLiteral, PreprocessingErrorMessage> {
        Ok(ast::KeywordArgumentTypeLiteral {
            name: t.name,
            arg_type: Box::new(LocTypeLiteral::preprocess(*t.arg_type)?)
        })
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

impl TypeLiteral {
    fn preprocess(t: Self) -> Result<ast::TypeLiteral, PreprocessingErrorMessage> {
        match t {
            TypeLiteral::Generic(s) => Ok(ast::TypeLiteral::Generic(s)),
            TypeLiteral::Void => Ok(ast::TypeLiteral::Void),
            TypeLiteral::Int => Ok(ast::TypeLiteral::Int),
            TypeLiteral::Bool => Ok(ast::TypeLiteral::Bool),
            TypeLiteral::Str => Ok(ast::TypeLiteral::Str),
            TypeLiteral::Tuple(type_vec) => {
                Ok(ast::TypeLiteral::Tuple(type_vec.into_iter().map(LocTypeLiteral::preprocess).collect::<Result<_,_>>()?))
            },
            TypeLiteral::List(t) => {
                Ok(ast::TypeLiteral::List(Box::new(LocTypeLiteral::preprocess(*t)?)))
            },
            TypeLiteral::Dict{ keys, values } => {
                Ok(ast::TypeLiteral::Dict {
                    keys: Box::new(LocTypeLiteral::preprocess(*keys)?),
                    values: Box::new(LocTypeLiteral::preprocess(*values)?)
                })
            },
            TypeLiteral::Callable{
                generics,
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type
            } => {
                Ok(ast::TypeLiteral::Callable {
                    generics,
                    positional_arguments: positional_arguments.into_iter().map(LocTypeLiteral::preprocess).collect::<Result<_,_>>()?,
                    variadic_argument: Box::new(variadic_argument.map(LocTypeLiteral::preprocess).transpose()?),
                    keyword_arguments: keyword_arguments.into_iter().map(KeywordArgumentTypeLiteral::preprocess).collect::<Result<_,_>>()?,
                    keyword_variadic_argument: Box::new(keyword_variadic_argument.map(LocTypeLiteral::preprocess).transpose()?),
                    return_type: Box::new(LocTypeLiteral::preprocess(*return_type)?)
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocTypeLiteral {
    pub expr: TypeLiteral,
    pub loc: Loc
}

impl LocTypeLiteral {
    fn preprocess(t: Self) -> Result<ast::LocTypeLiteral, PreprocessingErrorMessage> {
        Ok(ast::LocTypeLiteral {
            expr: TypeLiteral::preprocess(t.expr)?,
            loc: t.loc
        })
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
    Or
}

impl BinOp {
    fn preprocess(op: Self) -> Result<ast::BinOp, PreprocessingErrorMessage> {
        Ok(match op {
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
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not
}

impl UnOp {
    fn preprocess(op: Self) -> Result<ast::UnOp, PreprocessingErrorMessage> {
        Ok(match op {
            UnOp::Neg => ast::UnOp::Neg,
            UnOp::Not => ast::UnOp::Not,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LocCallArgument {
    pub argument: CallArgument,
    pub loc: Range<usize>
}

#[derive(Debug, Clone)]
pub enum CallArgument {
    PositionalArgument(LocExpr),
    Variadic(LocExpr),
    KeywordArgument(String, LocExpr),
    KeywordVariadic(LocExpr)
}


#[derive(Debug, Clone)]
pub struct LambdaArgument {
    pub name: String,
    pub loc: Loc
}

impl LambdaArgument {
    fn preprocess(arg: Self) -> Result<ast::LambdaArgument, PreprocessingErrorMessage> {
        Ok(ast::LambdaArgument {
            name: arg.name,
            loc: arg.loc
        })
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
        arguments: Vec<LocCallArgument>
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

impl Expr {
    fn preprocess(e: Self) -> Result<ast::Expr, PreprocessingErrorMessage> {
        match e {
            Expr::Variable(s) => Ok(ast::Expr::Variable(s)),
            Expr::DotAccess(e, v) => Ok(ast::Expr::DotAccess(Box::new(LocExpr::preprocess(*e)?), v)),
            Expr::Int(i) => Ok(ast::Expr::Int(i)),
            Expr::Bool(b) => Ok(ast::Expr::Bool(b)),
            Expr::Str(s) => Ok(ast::Expr::Str(s)),
            Expr::Tuple(v) => {
                Ok(ast::Expr::Tuple(v.into_iter().map(LocExpr::preprocess).collect::<Result<_,_>>()?))
            },
            Expr::List(v) => {
                Ok(ast::Expr::List(v.into_iter().map(LocExpr::preprocess).collect::<Result<_,_>>()?))
            },
            Expr::Dictionary(v) => {
                Ok(ast::Expr::Dictionary(
                    v.into_iter()
                     .map(|(k, v)| Ok((LocExpr::preprocess(k)?, LocExpr::preprocess(v)?)))
                     .collect::<Result<_,_>>()?
                ))
            },
            Expr::BinOp { op, left, right } => {
                Ok(ast::Expr::BinOp {
                    op: BinOp::preprocess(op)?,
                    left: Box::new(LocExpr::preprocess(*left)?),
                    right: Box::new(LocExpr::preprocess(*right)?)
                })
            },
            Expr::UnOp { op, expr } => {
                Ok(ast::Expr::UnOp {
                    op: UnOp::preprocess(op)?,
                    expr: Box::new(LocExpr::preprocess(*expr)?)
                })
            },
            Expr::FunctionCall {
                function,
                arguments
            } => {
                let mut positional_arguments: Vec<ast::CallArgument> = Vec::new();
                let mut variadic_argument: Option<ast::CallArgument> = None;
                let mut keyword_arguments: Vec<ast::CallKeywordArgument> = Vec::new();
                let mut keyword_variadic_argument: Option<ast::CallArgument> = None;
            
                for arg in arguments {

                    match arg.argument {
                        CallArgument::PositionalArgument(expr) => {
                            if !variadic_argument.is_none() || !keyword_arguments.is_empty() || !keyword_variadic_argument.is_none() {
                                return Err(PreprocessingErrorMessage {
                                    error: PreprocessingError::FunctionProcessingError(
                                        String::from("Unexpected Positional Argument")
                                    ),
                                    loc: Some(arg.loc),
                                });
                            }
            
                            positional_arguments.push(ast::CallArgument {
                                expr: Box::new(LocExpr::preprocess(expr)?),
                                loc: arg.loc.clone(),
                            });
                        }
                        CallArgument::Variadic(expr) => {
                            if !keyword_arguments.is_empty() || !keyword_variadic_argument.is_none() {
                                return Err(PreprocessingErrorMessage {
                                    error: PreprocessingError::FunctionProcessingError(
                                        String::from("Unexpected Variadic Argument")
                                    ),
                                    loc: Some(arg.loc),
                                });
                            }
            
                            if variadic_argument.is_none() {
                                variadic_argument = Some(ast::CallArgument {
                                    expr: Box::new(LocExpr::preprocess(expr)?),
                                    loc: arg.loc.clone(),
                                });
                            } else {
                                return Err(PreprocessingErrorMessage {
                                    error: PreprocessingError::FunctionProcessingError(
                                        String::from("Duplicate variadic argument")
                                    ),
                                    loc: Some(arg.loc),
                                });
                            }
                        }
                        CallArgument::KeywordArgument(name, expr) => {
                            if !keyword_variadic_argument.is_none() {
                                return Err(PreprocessingErrorMessage {
                                    error: PreprocessingError::FunctionProcessingError(
                                        String::from("Unexpected Keyword Argument")
                                    ),
                                    loc: Some(arg.loc),
                                });
                            }

                            match keyword_arguments.iter().filter(|x| x.name == name).collect::<Vec<&ast::CallKeywordArgument>>().get(0) {
                                Some(other_arg) => {
                                    return Err(PreprocessingErrorMessage {
                                        error: PreprocessingError::FunctionProcessingError(
                                            format!("Keyword Argument with name {} already exists", other_arg.name)
                                        ),
                                        loc: Some(other_arg.loc.start..arg.loc.end),
                                    });
                                }
                                _ => ()
                            }
            
                            keyword_arguments.push(ast::CallKeywordArgument {
                                name: name.clone(),
                                expr: Box::new(LocExpr::preprocess(expr)?),
                                loc: arg.loc.clone(),
                            });
                        }
                        CallArgument::KeywordVariadic(expr) => {
                            if keyword_variadic_argument.is_none() {
                                keyword_variadic_argument = Some(ast::CallArgument {
                                    expr: Box::new(LocExpr::preprocess(expr)?),
                                    loc: arg.loc.clone(),
                                });
                            } else {
                                return Err(PreprocessingErrorMessage {
                                    error: PreprocessingError::FunctionProcessingError(
                                        String::from("Duplicate keyword variadic argument")
                                    ),
                                    loc: Some(arg.loc),
                                });
                            }
                        }
                    }
                }

                Ok(ast::Expr::FunctionCall {
                    function: Box::new(LocExpr::preprocess(*function)?),
                    positional_arguments: positional_arguments,
                    variadic_argument: variadic_argument,
                    keyword_arguments: keyword_arguments,
                    keyword_variadic_argument: keyword_variadic_argument
                })
            },
            Expr::Indexing { indexed, indexer } => {
                Ok(ast::Expr::Indexing {
                    indexed: Box::new(LocExpr::preprocess(*indexed)?),
                    indexer: Box::new(LocExpr::preprocess(*indexer)?)
                })
            },
            Expr::FunctionPtr(s) => Ok(ast::Expr::FunctionPtr(s)),
            Expr::Lambda { arguments, expr } => {
                Ok(ast::Expr::Lambda {
                    arguments: arguments.into_iter().map(LambdaArgument::preprocess).collect::<Result<_,_>>()?,
                    expr: Box::new(LocExpr::preprocess(*expr)?)
                })
            },
            Expr::Block { statements } => {
                Ok(ast::Expr::Block {
                    statements: statements.into_iter().map(LocStmt::preprocess).collect::<Result<_,_>>()?
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocExpr {
    pub expr: Expr,
    pub loc: Loc
}

impl LocExpr {

    fn preprocess(le: Self) -> Result<ast::LocExpr, PreprocessingErrorMessage> {
        Ok(ast::LocExpr {
            expr: Expr::preprocess(le.expr)?,
            loc: le.loc,
            typ: ast::Type::Unknown
        })
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
    For {
        pattern: LocExpr,
        iterable: LocExpr,
        body: Box<LocStmt>
    },
    Block {
        statements: Vec<LocStmt> 
    },
    Expression {
        expr: LocExpr
    }
}

impl Stmt {
    fn preprocess(s: Self) -> Result<ast::Stmt, PreprocessingErrorMessage> {
        match s {
            Stmt::Assignment { target, expr: expression } => {
                Ok(ast::Stmt::Assignment {
                    target: LocExpr::preprocess(target)?,
                    expr: LocExpr::preprocess(expression)?
                })
            },
            Stmt::FunctionCall { expr: expression } => {
                Ok(ast::Stmt::FunctionCall {
                    expr: LocExpr::preprocess(expression)?
                })
            },
            Stmt::Return { expr: expression } => {
                Ok(ast::Stmt::Return {
                    expr: LocExpr::preprocess(expression)?
                })
            },
            Stmt::IfElse { cond: condition, if_body, else_body } => {
                let loc = condition.clone().loc;
                let func_call_cond = LocExpr {
                    loc: loc.clone(),
                    expr: Expr::FunctionCall {
                        function: Box::new(LocExpr {
                            expr: Expr::Variable(String::from("bool")),
                            loc: loc.clone(),
                        }),
                        arguments: vec![
                            LocCallArgument {
                                argument: CallArgument::PositionalArgument(condition),
                                loc: loc.clone(),
                            }
                        ],
                    }
                };
                
                Ok(ast::Stmt::IfElse {
                    cond: LocExpr::preprocess(func_call_cond)?, 
                    if_body: Box::new(LocStmt::preprocess(*if_body)?),
                    else_body: Box::new(LocStmt::preprocess(*else_body)?)
                })
            },
            Stmt::While { cond: condition, body } => {
                let loc = condition.clone().loc;
                let func_call_cond = LocExpr {
                    loc: condition.loc.clone(),
                    expr: Expr::FunctionCall {
                        function: Box::new(LocExpr {
                            expr: Expr::Variable(String::from("bool")),
                            loc: condition.loc.clone(),
                        }),
                        arguments: vec![
                            LocCallArgument {
                                argument: CallArgument::PositionalArgument(condition),
                                loc: loc.clone(),
                            }
                        ],
                    }
                };

                Ok(ast::Stmt::While {
                    cond: LocExpr::preprocess(func_call_cond)?,
                    body: Box::new(LocStmt::preprocess(*body)?)
                })
            },
            Stmt::For { pattern, iterable, body } => {
                let loop_counter = get_unique_var();
                let init_counter = LocStmt {
                    stmt: Stmt::Assignment { 
                        target: LocExpr {expr: Expr::Variable(loop_counter.clone()), loc: pattern.loc.clone()}, 
                        expr: LocExpr {expr: Expr::Int(0), loc: pattern.loc.clone()}},
                    loc: pattern.loc.clone(),
                };
                let increase_counter = LocStmt {
                    stmt: Stmt::Assignment { 
                        target: LocExpr {expr: Expr::Variable(loop_counter.clone()), loc: pattern.loc.clone()}, 
                        expr: LocExpr {
                            expr: Expr::BinOp {
                            op: BinOp::Add,
                            left: Box::new(LocExpr {
                                expr: Expr::Variable(loop_counter.clone()),
                                loc: pattern.loc.clone()
                            }),
                            right: Box::new(LocExpr {
                                expr: Expr::Int(1),
                                loc: pattern.loc.clone()
                            })
                            },
                            loc: pattern.loc.clone()
                        } 
                    },
                    loc: pattern.loc.clone()
                };
                let iterable_len = LocExpr {
                    expr: Expr::FunctionCall { 
                        function: Box::new(LocExpr {
                            expr: Expr::Variable("len".to_string()),
                            loc: iterable.loc.clone()
                        }),
                        arguments: vec![LocCallArgument {
                            argument: CallArgument::PositionalArgument(iterable.clone()),
                            loc: iterable.loc.clone()
                        }]
                    },
                    loc: iterable.loc.clone()
                };
                let condition = LocExpr {
                    expr: Expr::BinOp {
                        op: BinOp::Lt,
                        left: Box::new(LocExpr {
                        expr: Expr::Variable(loop_counter.clone()),
                        loc: iterable.loc.clone()
                        }),
                        right: Box::new(iterable_len.clone())
                    },
                    loc: iterable.loc.clone()
                };
                let pattern_loc = pattern.loc.clone();
                let set_element = LocStmt {
                    stmt: Stmt::Assignment { 
                        target: pattern, 
                        expr: LocExpr {
                            expr: Expr::Indexing {
                                indexed: Box::new(iterable.clone()),
                                indexer: Box::new(LocExpr {expr: Expr::Variable(loop_counter.clone()), loc: pattern_loc.clone()}) 
                            },
                            loc: pattern_loc.clone()
                        } 
                    },
                    loc: pattern_loc
                };

                let body_loc = body.loc.clone();

                let new_body = LocStmt {
                    stmt: Stmt::Block { 
                        statements: vec![
                            set_element,
                            *body,
                            increase_counter
                        ]
                    },
                    loc: body_loc.clone()
                };

                let new_loop = LocStmt {
                    stmt: Stmt::While { cond: condition, body: Box::new(new_body) },
                    loc: body_loc
                };               

                Ok(Stmt::preprocess(Stmt::Block { 
                    statements: vec![
                        init_counter,
                        new_loop
                    ]
                })?)
            },
            Stmt::Block { statements } => {
                Ok(ast::Stmt::Block {
                    statements: statements.into_iter().map(LocStmt::preprocess).collect::<Result<_,_>>()?
                })
            },
            Stmt::Expression { expr: expression } => {
                Ok(ast::Stmt::Expression {
                    expr: LocExpr::preprocess(expression)?
                })
            }
        }
    }
}


#[derive(Debug, Clone)]
pub struct LocStmt {
    pub stmt: Stmt,
    pub loc: Loc 
}

impl LocStmt {
    fn preprocess(ls: Self) -> Result<ast::LocStmt, PreprocessingErrorMessage> {
        Ok(ast::LocStmt {
            stmt: Stmt::preprocess(ls.stmt)?,
            loc: ls.loc
        })
    }
}



#[derive(Debug, Clone)]
pub enum Argument {
    PositionalArgument(String),
    Variadic(String),
    KeywordArgument(String, LocExpr),
    KeywordVariadic(String)
}


#[derive(Debug, Clone)]
pub struct LocArgument {
    pub argument: Argument,
    pub loc: Range<usize>
}


#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<LocArgument>,
    pub body: Box<LocStmt>,
    pub loc: Range<usize>
}


impl Function {
    fn preprocess(func: Self) -> Result<ast::Function, PreprocessingErrorMessage> {
        let mut positional_arguments: Vec<ast::Argument> = Vec::new();
        let mut variadic_argument: Option<ast::Argument> = None;
        let mut keyword_arguments: Vec<ast::KeywordArgument> = Vec::new();
        let mut keyword_variadic_argument: Option<ast::Argument> = None;

        for arg in func.arguments {

            let name = match arg.argument.clone() {
                Argument::PositionalArgument(name)
                | Argument::Variadic(name)
                | Argument::KeywordArgument(name, _)
                | Argument::KeywordVariadic(name)
                => name,
            };

            match positional_arguments.iter().filter(|x| x.name == name).collect::<Vec<&ast::Argument>>().get(0) {
                Some(other_arg) => {
                    return Err(PreprocessingErrorMessage {
                        error: PreprocessingError::FunctionProcessingError(
                            format!("Argument with name {} already exists", other_arg.name)
                        ),
                        loc: Some(other_arg.loc.start..arg.loc.end),
                    });
                },
                _ => ()
            }

            match variadic_argument.as_ref().or_else(|| None) {
                Some(other_arg) => {
                    return Err(PreprocessingErrorMessage {
                        error: PreprocessingError::FunctionProcessingError(
                            format!("Argument with name {} already exists", other_arg.name)
                        ),
                        loc: Some(other_arg.loc.start..arg.loc.end),
                    });
                },
                _ => {}
            }

            match keyword_arguments.iter().filter(|x| x.name == name).collect::<Vec<&ast::KeywordArgument>>().get(0) {
                Some(other_arg) => {
                    return Err(PreprocessingErrorMessage {
                        error: PreprocessingError::FunctionProcessingError(
                            format!("Argument with name {} already exists", other_arg.name)
                        ),
                        loc: Some(other_arg.loc.start..arg.loc.end),
                    });
                },
                _ => ()
            }

            match &keyword_variadic_argument.as_ref().or_else(|| None) {
                Some(other_arg) => {
                    return Err(PreprocessingErrorMessage {
                        error: PreprocessingError::FunctionProcessingError(
                            format!("Argument with name {} already exists", other_arg.name)
                        ),
                        loc: Some(other_arg.loc.start..arg.loc.end),
                    });
                },
                _ => {}
            }
            
            match arg.argument {
                Argument::PositionalArgument(name) => {
                    if !variadic_argument.is_none() || !keyword_arguments.is_empty() || !keyword_variadic_argument.is_none() {
                        return Err(PreprocessingErrorMessage {
                            error: PreprocessingError::FunctionProcessingError(
                                String::from("Unexpected Positional Argument")
                            ),
                            loc: Some(arg.loc),
                        });
                    }

                    positional_arguments.push(ast::Argument {
                        name: name.clone(),
                        arg_type: None,
                        loc: arg.loc.clone(),
                    });
                }
                Argument::Variadic(name) => {
                    if !keyword_arguments.is_empty() || !keyword_variadic_argument.is_none() {
                        return Err(PreprocessingErrorMessage {
                            error: PreprocessingError::FunctionProcessingError(
                                String::from("Unexpected Variadic Argument")
                            ),
                            loc: Some(arg.loc),
                        });
                    }

                    if variadic_argument.is_none() {
                        variadic_argument = Some(ast::Argument {
                            name: name.clone(),
                            arg_type: None,
                            loc: arg.loc.clone(),
                        });
                    } else {
                        return Err(PreprocessingErrorMessage {
                            error: PreprocessingError::FunctionProcessingError(
                                String::from("Duplicate variadic argument")
                            ),
                            loc: Some(arg.loc),
                        });
                    }
                }
                Argument::KeywordArgument(name, expression) => {
                    if !keyword_variadic_argument.is_none() {
                        return Err(PreprocessingErrorMessage {
                            error: PreprocessingError::FunctionProcessingError(
                                String::from("Unexpected Keyword Argument")
                            ),
                            loc: Some(arg.loc),
                        });
                    }

                    keyword_arguments.push(ast::KeywordArgument {
                        name: name.clone(),
                        expr: LocExpr::preprocess(expression.clone())?,
                        loc: arg.loc.clone(),
                    });
                }
                Argument::KeywordVariadic(name) => {
                    if keyword_variadic_argument.is_none() {
                        keyword_variadic_argument = Some(ast::Argument {
                            name: name.clone(),
                            arg_type: None,
                            loc: arg.loc.clone(),
                        });
                    } else {
                        return Err(PreprocessingErrorMessage {
                            error: PreprocessingError::FunctionProcessingError(
                                String::from("Duplicate keyword variadic argument")
                            ),
                            loc: Some(arg.loc),
                        });
                    }
                }
            }
        }

        Ok(ast::Function {
            name: func.name.clone(),
            contract: ast::FunctionPrototype {
                positional_arguments,
                variadic_argument,
                keyword_arguments,
                keyword_variadic_argument,
                return_type: None
            },
            body: Box::new(LocStmt::preprocess(*func.body)?),
            loc: func.loc,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: HashMap<String, Function>
}

impl Program {
    pub fn preprocess(prog: Self) -> Result<ast::Program, PreprocessingErrorMessage> {
        Ok(ast::Program {
            functions: prog.functions
                            .into_iter()
                            .map(|(name, func)| Ok((name, Function::preprocess(func)?)))
                            .collect::<Result<_,_>>()?
        })
    }
}