use std::collections::{HashMap, HashSet};

use crate::{ast::{self, FunctionPrototype, GenericLiteral, Type}, interpreter::InterpreterErrorMessage};

pub enum TypecheckingError {
    MissingTypeAnnotation,
    MissingReturnTypeAnnotation,
    UnknownType(String),
    NotExpectedType(ast::Expr, ast::Type),
    NotSubtype(ast::Type, ast::Type),
    NotAValidLocation(ast::Expr),
    NotTuple(ast::Expr),
    UnpackCountMismatch(usize, Vec<ast::LocExpr>, usize, Vec<Type>),
    NeedsToBeVariable(ast::Expr)
}

pub struct TypecheckingErrorMessage {
    error: TypecheckingError,
    loc: ast::Loc
}

impl ast::Type {
    
    pub fn subtypes(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }

        match (self, other) {
            (_, ast::Type::Unknown) => true,
            (ast::Type::Unknown, _) => false,
            (ast::Type::Tuple(a), ast::Type::Tuple(b)) => a.iter().zip(b).all(|(a, b)| a.subtypes(b)),
            (ast::Type::List(a), ast::Type::List(b)) => a.subtypes(b),
            (ast::Type::Dict { keys: keys_a, values: values_a }, ast::Type::Dict { keys: keys_b, values: values_b }) => keys_a.subtypes(keys_b) && values_a.subtypes(values_b),
                (ast::Type::Callable { generics, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, return_type }, _) => todo!(),
            _ => false
        }
    }
}

impl ast::TypeLiteral {

    pub fn validate_generics(&self, generics: &Vec<String>, loc: &ast::Loc) -> Result<(), TypecheckingErrorMessage> {
        match self {
            ast::TypeLiteral::Generic(s) => {
                if !generics.contains(&s) {
                    Err(TypecheckingErrorMessage {
                        error: TypecheckingError::UnknownType(s.clone()),
                        loc: loc.clone(),
                    })
                } else {
                    Ok(())
                }
            },
            ast::TypeLiteral::Tuple(tys) => tys.iter().map(|t| Self::validate_generics(&t.typ, generics, loc)).collect(),
            ast::TypeLiteral::List(t) => Self::validate_generics(&t.typ, generics, loc),
            ast::TypeLiteral::Dict{keys, values} => {Self::validate_generics(&keys.typ, generics, loc)?; Self::validate_generics(&values.typ, generics, loc)},
            ast::TypeLiteral::Callable { generics, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, return_type } => todo!(),
            _ => Ok(())
        }
    }

    pub fn get_type(&self) -> ast::Type {
        match self {
            ast::TypeLiteral::Generic(s) => ast::Type::Generic(s.clone()),
            ast::TypeLiteral::Void => ast::Type::Unit,
            ast::TypeLiteral::Int => ast::Type::Int,
            ast::TypeLiteral::Bool => ast::Type::Bool,
            ast::TypeLiteral::Str => ast::Type::Str,
            ast::TypeLiteral::Tuple(v) => ast::Type::Tuple(v.iter().map(|ltl| Self::get_type(&ltl.typ)).collect()),
            ast::TypeLiteral::List(t) => ast::Type::List(Box::new(Self::get_type(&t.typ))),
            ast::TypeLiteral::Dict { keys, values } => ast::Type::Dict { keys: Box::new(Self::get_type(&keys.typ)), values: Box::new(Self::get_type(&values.typ)) },
            ast::TypeLiteral::Callable { generics, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, return_type } => todo!(),
        }
    }
}


#[derive(Clone)]
struct ProgramEnv {
    functions: HashMap<String, ast::Type>
}


impl ast::Program {

    pub fn verify(self) -> Result<(), TypecheckingErrorMessage> {
        todo!()
    }

    pub fn typecheck(self) -> Result<Self, TypecheckingErrorMessage> {

        let mut function_type_mapping = HashMap::new();

        let prototype_res: Result<Vec<(String, ast::Function)>, TypecheckingErrorMessage> = self.functions.into_iter().map(|(s,f)| {
            match f.contract.typecheck(&f.loc) {
                Ok(p) => {function_type_mapping.insert(s.clone(), p.typ.clone()); Ok((s, ast::Function {name: f.name, contract: p, body: f.body, loc: f.loc}))},
                Err(e) => Err(e)
            }
        }).collect();

        let env = ProgramEnv {
            functions: function_type_mapping
        };

        let res: Result<HashMap<String, ast::Function>, TypecheckingErrorMessage> = prototype_res?.into_iter().map(|(s,f)| {
            match f.typecheck(&env) {
                Ok(f) => Ok((s, f)),
                Err(e) => Err(e)
            }
        }).collect();


        Ok(ast::Program {functions: res?})
    }

}


impl ast::FunctionPrototype {

    pub fn verify(self) -> Result<(), TypecheckingErrorMessage> {
        todo!()
    }

    pub fn typecheck(self, loc: &ast::Loc) -> Result<Self, TypecheckingErrorMessage> {
        let generics: Vec<String> = self.generics.iter().map(|x| x.name.clone()).collect();

        let positional_arguments: Vec<ast::Argument> = self.positional_arguments
            .into_iter()
            .map(|arg| {
                let type_lit = arg.arg_type_literal
                    .ok_or_else(|| TypecheckingErrorMessage {
                        error: TypecheckingError::MissingTypeAnnotation,
                        loc: arg.loc.clone(),
                    })?;

                type_lit.typ.validate_generics(&generics, &arg.loc)?;
                let typ = type_lit.typ.get_type();

                Ok(ast::Argument {
                    name: arg.name,
                    arg_type_literal: Some(type_lit),
                    loc: arg.loc,
                    typ: typ
                })
            })
            .collect::<Result<_, _>>()?;

        let variadic_argument: Option<ast::Argument> = match self.variadic_argument {
            Some(var) => {
                let type_lit = var.arg_type_literal
                    .ok_or_else(|| TypecheckingErrorMessage {
                        error: TypecheckingError::MissingTypeAnnotation,
                        loc: var.loc.clone(),
                    })?;

                type_lit.typ.validate_generics(&generics, &var.loc)?;
                let typ = type_lit.typ.get_type();
                
                let expected = ast::Type::List(Box::new(ast::Type::Unknown));
                if !typ.subtypes(&expected) {
                    return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotSubtype(typ.clone(), expected),
                        loc: var.loc.clone()
                    })
                }

                Some(ast::Argument {
                    name: var.name,
                    arg_type_literal: Some(type_lit),
                    loc: var.loc,
                    typ: typ
                })
            },
            _ => None
        };

        let keyword_arguments: Vec<ast::KeywordArgument> = self.keyword_arguments
            .into_iter()
            .map(|arg| {
                let final_type = match &arg.arg_type_literal {
                    Some(lit) => {
                        lit.typ.validate_generics(&generics, &arg.loc)?;
                        let ann = lit.typ.get_type();
                        if !arg.expr.clone().check(&ann)? {
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::NotExpectedType(arg.expr.expr.clone(), ann),
                                loc: arg.loc.clone(),
                            });
                        }
                        ann
                    },
                    _ => arg.expr.infer()?
                        .ok_or_else(|| TypecheckingErrorMessage {
                            error: TypecheckingError::MissingTypeAnnotation,
                            loc: arg.loc.clone(),
                        })?
                };

                Ok(ast::KeywordArgument {
                    name: arg.name,
                    expr: arg.expr,
                    arg_type_literal: arg.arg_type_literal,
                    loc: arg.loc,
                    typ: final_type,
                })
            })
            .collect::<Result<_, _>>()?;

        let keyword_variadic_argument: Option<ast::Argument> = match self.keyword_variadic_argument {
            Some(var) => {
                let type_lit = var.arg_type_literal
                    .ok_or_else(|| TypecheckingErrorMessage {
                        error: TypecheckingError::MissingTypeAnnotation,
                        loc: var.loc.clone(),
                    })?;

                type_lit.typ.validate_generics(&generics, &var.loc)?;
                let typ = type_lit.typ.get_type();
                
                let expected = ast::Type::Dict{keys: Box::new(ast::Type::Str), values: Box::new(ast::Type::Unknown)};
                if !typ.subtypes(&expected) {
                    return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotSubtype(typ.clone(), expected),
                        loc: var.loc.clone()
                    })
                }

                Some(ast::Argument {
                    name: var.name,
                    arg_type_literal: Some(type_lit),
                    loc: var.loc,
                    typ: typ
                })
            },
            _ => None
        };

        let return_typ_literal = self.return_type_literal
            .ok_or_else(|| TypecheckingErrorMessage {
                error: TypecheckingError::MissingReturnTypeAnnotation,
                loc: loc.clone(),
            })?;

        return_typ_literal.typ.validate_generics(&generics, &return_typ_literal.loc)?;
        let return_typ = return_typ_literal.typ.get_type();

        let typ = ast::Type::Callable { 
            generics, 
            positional_arguments: positional_arguments.iter().map(|arg| arg.typ.clone()).collect(),
            variadic_argument: variadic_argument.clone().map(|arg| Box::new(arg.typ)),
            keyword_arguments: keyword_arguments.iter().map(|arg| ast::KeywordArgumentType {name: arg.name.clone(), arg_type: arg.typ.clone()}).collect(),
            keyword_variadic_argument: keyword_variadic_argument.clone().map(|arg| Box::new(arg.typ)),
            return_type: Box::new(return_typ.clone()) 
        };


        Ok(FunctionPrototype {
            generics: self.generics,
            positional_arguments,
            variadic_argument,
            keyword_arguments,
            keyword_variadic_argument,
            return_type_literal: Some(return_typ_literal),
            return_typ,
            typ: typ.clone()
        })
    }
}


impl ast::Function {

    pub fn verify(self) -> Result<(), TypecheckingErrorMessage> {
        todo!()
    }

    pub fn typecheck(self, env: &ProgramEnv) -> Result<Self, TypecheckingErrorMessage> {
        let function_typ = self.contract.typ;

        todo!()
    }

}

#[derive(Clone)]
struct FunctionEnv {
    program_env: ProgramEnv,
    return_type: ast::Type,
    variable_types: Vec<HashMap<String, ast::Type>> // stack because we can enter blocks
}

enum InsertVariableResult {
    ExistsSame,
    Subtypes,
    DoesNotExist
}

impl FunctionEnv {

    pub fn insert_variable_type(&mut self, var: &String, typ: &ast::Type, loc: &ast::Loc) -> Result<InsertVariableResult, TypecheckingErrorMessage> {
        for mapping in self.variable_types.iter() {
            if let Some(existing_typ) = mapping.get(var) {
                if typ == existing_typ {
                    return Ok(InsertVariableResult::ExistsSame);
                } else if typ.subtypes(existing_typ) {
                    self.variable_types.last_mut().unwrap().insert(var.clone(), typ.clone());
                    return Ok(InsertVariableResult::Subtypes);
                } else {
                    return Err(TypecheckingErrorMessage {error: TypecheckingError::NotSubtype(typ.clone(), existing_typ.clone()), loc: loc.clone()})
                }
            }
        }

        self.variable_types.last_mut().unwrap().insert(var.clone(), typ.clone());

        return Ok(InsertVariableResult::DoesNotExist);
    }

    pub fn get_variable_type(&mut self, var: &String) -> Option<ast::Type> {
        for mapping in self.variable_types.iter() {
            if let Some(existing_typ) = mapping.get(var) {
                return Some(existing_typ.clone());
            }
        }
        return None;
    }

}

impl ast::LocStmt {
    pub fn verify(self) -> Result<(), TypecheckingErrorMessage> {
        todo!()
    }

    pub fn typecheck(self, env: &mut FunctionEnv) -> Result<Self, TypecheckingErrorMessage> {

        match self.stmt {
            ast::Stmt::Assignment { target, expr } => {
                let expr = expr.typecheck(env)?;
                match target.expr {
                    ast::Expr::Variable(ref var) => {
                        let mut new_env = env.clone();
                        new_env.insert_variable_type(var, &expr.typ, &expr.loc)?;
                        Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { target, expr }, loc: self.loc, typ: ast::Type::Unit})
                    },
                    ast::Expr::Indexing { indexed, indexer } => {
                        let indexed  = indexed.typecheck(env)?;
                        

                        if let ast::Type::Impossible = indexed.typ {
                            return Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                        target: ast::LocExpr {
                                            expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: indexer },
                                            loc: target.loc, typ: ast::Type::Impossible
                                        },
                                        expr: expr
                                    },
                                    loc: self.loc,
                                    typ: ast::Type::Impossible
                                });
                        }

                        match indexed.typ {
                            ast::Type::Generic(s) => todo!("Deep sigh..."),
                            ast::Type::List(ref element_type) => {
                                let indexer = indexer.typecheck(env)?;

                                if let ast::Type::Impossible = indexer.typ {
                                    return Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                                target: ast::LocExpr {
                                                    expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                                    loc: target.loc, typ: ast::Type::Impossible
                                                },
                                                expr: expr 
                                            },
                                            loc: self.loc,
                                            typ: ast::Type::Impossible
                                        });
                                }

                                if !indexer.typ.subtypes(&ast::Type::Int) {
                                    return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::NotExpectedType(indexer.expr.clone(), ast::Type::Int),
                                        loc: indexer.loc
                                    })
                                }

                                if !expr.typ.subtypes(&element_type) {
                                    return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::NotExpectedType(expr.expr.clone(), *element_type.clone()),
                                        loc: expr.loc
                                    })
                                }

                                let indexed_typ = indexed.typ.clone();
                                

                                Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                        target: ast::LocExpr {
                                            expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                            loc: target.loc, typ: indexed_typ
                                        },
                                        expr: expr 
                                    },
                                    loc: self.loc,
                                    typ: ast::Type::Unit
                                })
                            },
                            ast::Type::Dict{ref keys, ref values} => {
                                let indexer = indexer.typecheck(env)?;

                                if let ast::Type::Impossible = indexer.typ {
                                    return Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                                target: ast::LocExpr {
                                                    expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                                    loc: target.loc, typ: ast::Type::Impossible
                                                },
                                                expr: expr 
                                            },
                                            loc: self.loc,
                                            typ: ast::Type::Impossible
                                        });
                                }

                                if !indexer.typ.subtypes(&keys) {
                                    return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::NotExpectedType(indexer.expr.clone(), *keys.clone()),
                                        loc: indexer.loc
                                    })
                                }

                                if !expr.typ.subtypes(&values) {
                                    return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::NotExpectedType(expr.expr.clone(), *values.clone()),
                                        loc: expr.loc
                                    })
                                }

                                let indexed_typ = indexed.typ.clone();

                                Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                        target: ast::LocExpr {
                                            expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                            loc: target.loc, typ: indexed_typ
                                        },
                                        expr: expr 
                                    },
                                    loc: self.loc,
                                    typ: ast::Type::Unit
                                })
                            },
                            _ => return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::NotAValidLocation(expr.expr),
                                loc: expr.loc
                            })
                        }
                    },
                    ast::Expr::Tuple(ref elements) => {
                        
                        match expr.typ {
                            ast::Type::Tuple(ref unpacked_elements_types) => {
                                if elements.len() != unpacked_elements_types.len() {
                                    return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::UnpackCountMismatch(elements.len(), elements.clone(), unpacked_elements_types.len(), unpacked_elements_types.clone()),
                                        loc: expr.loc
                                    })
                                }

                                for (el, resulting_type) in elements.iter().zip(unpacked_elements_types) {
                                    match &el.expr {
                                        ast::Expr::Variable(x) => {
                                            env.insert_variable_type(x, &resulting_type, &el.loc);
                                        }
                                        _ => return Err(TypecheckingErrorMessage {
                                            error: TypecheckingError::NeedsToBeVariable(el.expr.clone()),
                                            loc: expr.loc
                                        })
                                    }
                                }

                                Ok(ast::LocStmt {stmt: ast::Stmt::Assignment { 
                                        target: target,
                                        expr: expr 
                                    },
                                    loc: self.loc,
                                    typ: ast::Type::Unit
                                })
                            },
                            _ => return Err(TypecheckingErrorMessage {
                                        error: TypecheckingError::NotTuple(expr.expr.clone()),
                                        loc: expr.loc
                                    })
                        }
                    },
                    _ => return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotAValidLocation(target.expr.clone()),
                        loc: target.loc.clone()
                    })
                }
            },
            ast::Stmt::FunctionCall { expr } => {
                let expr = expr.typecheck(env)?;
                Ok(ast::LocStmt {
                    stmt: ast::Stmt::FunctionCall { expr },
                    loc: self.loc,
                    typ: ast::Type::Impossible
                })
            },
            ast::Stmt::Return { expr } => {
                let expr = expr.typecheck(env)?;

                if !expr.typ.subtypes(&env.return_type) {
                    return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotExpectedType(expr.expr, env.return_type.clone()),
                        loc: expr.loc
                    })
                }

                return Ok(ast::LocStmt {
                    stmt: ast::Stmt::Return { expr: expr },
                    loc: self.loc,
                    typ: ast::Type::Impossible
                })
            },
            ast::Stmt::IfElse { cond, if_body, else_body } => todo!(),
            ast::Stmt::While { cond, body } => todo!(),
            ast::Stmt::Block { statements } => todo!(),
            ast::Stmt::SoftBlock { statements } => todo!(),
            ast::Stmt::Expression { expr } => todo!(),
            ast::Stmt::Break => todo!(),
            ast::Stmt::Continue => todo!(),
        }
    }
}


impl ast::LocExpr {

    pub fn typecheck(self, env: &FunctionEnv) -> Result<Self, TypecheckingErrorMessage> {
        todo!()
    }

    pub fn check(self, expected_type: &ast::Type) -> Result<bool, TypecheckingErrorMessage> {
        todo!()
    }

    pub fn infer(&self) -> Result<Option<ast::Type>, TypecheckingErrorMessage> {
        todo!()
    }
}