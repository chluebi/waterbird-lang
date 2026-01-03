use std::collections::{HashMap, HashSet};

use crate::ast;

pub enum TypecheckingError {
    MissingTypeAnnotation,
    MissingReturnTypeAnnotation,
    UnknownType(String),
    NotExpectedType(ast::Expr, ast::Type),
    NotSubtype(ast::Type, ast::Type),
    NotAValidLocation(ast::Expr),
    NotTuple(ast::Expr),
    UnpackCountMismatch(usize, Vec<ast::LocExpr>, usize, Vec<ast::Type>),
    NeedsToBeVariable(ast::Expr),
    Unreachable(),
    VariableNotFound(String),
    ArgumentsDontMatchFunction(ast::Type)
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
            (ast::Type::Impossible, _) => true,
            (_, ast::Type::Impossible) => false,
            (ast::Type::Tuple(a), ast::Type::Tuple(b)) => a.iter().zip(b).all(|(a, b)| a.subtypes(b)),
            (ast::Type::List(_), ast::Type::List(_)) => false, // evil variance
            (ast::Type::Dict { keys: _, values: _ }, ast::Type::Dict { keys: keys_b, values: values_b }) => false, // evil variance
            (ast::Type::Callable { 
                generics: generics_a,
                positional_arguments: positional_arguments_a,
                variadic_argument: variadic_argument_a,
                keyword_arguments: keyword_arguments_a,
                keyword_variadic_argument: keyword_variadic_argument_a,
                return_type: return_type_a 
            }, 
            ast::Type::Callable { 
                generics: generics_b,
                positional_arguments: positional_arguments_b,
                variadic_argument: variadic_argument_b,
                keyword_arguments: keyword_arguments_b,
                keyword_variadic_argument: keyword_variadic_argument_b,
                return_type: return_type_b
            }) => {
                
                if generics_a != generics_b {
                    todo!()
                }

                if !return_type_a.subtypes(return_type_b) {
                    return false;
                }

                // all arguments that B accepts must be accepted by A
                for (i, type_b) in positional_arguments_b.iter().enumerate() {
                    let type_a = if i < positional_arguments_a.len() {
                        // handled by normal a arguments
                        &positional_arguments_a[i]
                    } else if let Some(var_a) = variadic_argument_a {
                        // handled by variadic a
                        var_a
                    } else {
                        return false;
                    };

                    if !type_a.subtypes(type_b) {
                        return false;
                    }
                }

                // B has variadic
                if let Some(var_b) = variadic_argument_b {
                    // A has variadic as well, needs to be subtype
                    if let Some(var_a) = variadic_argument_a {
                        if !var_b.subtypes(var_a) { return false; }
                    } else {
                        // A cannot handle B variadic
                        return false; 
                    }
                    
                    // B variadic covers extra arguments of A
                    if positional_arguments_a.len() > positional_arguments_b.len() {
                        for type_a in &positional_arguments_a[positional_arguments_b.len()..] {
                            if !var_b.subtypes(type_a) { return false; }
                        }
                    }
                } else {
                    // A cannot ask for more arguments than B asks for
                    if positional_arguments_a.len() > positional_arguments_b.len() {
                        return false;
                    }
                }

                // A has to handle all keyword arguments that B accepts
                for kw_b in keyword_arguments_b {
                    let type_a = if let Some(kw_a) = keyword_arguments_a.iter().find(|k| k.name == kw_b.name) {
                        &kw_a.arg_type
                    } else if let Some(kv_a) = keyword_variadic_argument_a {
                        kv_a // kwargs
                    } else {
                        return false;
                    };

                    if !kw_b.arg_type.subtypes(type_a) {
                        return false;
                    }
                }

                // B kwargs -> A kwargs
                if let Some(kv_b) = keyword_variadic_argument_b {
                    if let Some(kv_a) = keyword_variadic_argument_a {
                         if !kv_b.subtypes(kv_a) { return false; }
                    } else {
                        return false;
                    }

                    // A extra args can be handled by B kwargs
                    for kw_a in keyword_arguments_a {
                        if !keyword_arguments_b.iter().any(|k| k.name == kw_a.name) {
                            if !kv_b.subtypes(&kw_a.arg_type) { return false; }
                        }
                    }
                } else {
                    // no kwargs in B, A cannot ask for more keywords
                    for kw_a in keyword_arguments_a {
                        if !keyword_arguments_b.iter().any(|k| k.name == kw_a.name) {
                            return false;
                        }
                    }
                }

                true
            },
            _ => false
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (_, ast::Type::Unknown) | (ast::Type::Unknown, _) => ast::Type::Unknown,
            (ast::Type::Impossible, x) | (x, ast::Type::Impossible) => x.clone(),
            (ast::Type::Tuple(a), ast::Type::Tuple(b)) => ast::Type::Tuple(a.iter().zip(b).map(|(a, b)| a.join(b)).collect()),
            (ast::Type::List(_), ast::Type::List(_)) => ast::Type::List(Box::new(ast::Type::Unknown)), // evil variance
            (ast::Type::Dict { keys: _, values: _ }, ast::Type::Dict { keys: _, values: _ }) => ast::Type::Dict { keys: Box::new(ast::Type::Unknown), values: Box::new(ast::Type::Unknown) },
            (ast::Type::Callable { 
                generics: generics_a,
                positional_arguments: positional_arguments_a,
                variadic_argument: variadic_argument_a,
                keyword_arguments: keyword_arguments_a,
                keyword_variadic_argument: keyword_variadic_argument_a,
                return_type: return_type_a 
            }, 
            ast::Type::Callable { 
                generics: generics_b,
                positional_arguments: positional_arguments_b,
                variadic_argument: variadic_argument_b,
                keyword_arguments: keyword_arguments_b,
                keyword_variadic_argument: keyword_variadic_argument_b,
                return_type: return_type_b 
            }) => {

                if generics_a != generics_b {
                    todo!()
                }

                // okay so basically the problem here is the following
                // we need to a find a function that both other functions subtype
                if self.subtypes(other) {
                    return other.clone();
                }

                if other.subtypes(self) {
                    return self.clone();
                }

                // if they do not subtype each other, we need to account for contravariance of return arguments, which means meets etc.
                // todo
                return ast::Type::Unknown;
            },
            _ => ast::Type::Unknown
        }
    }

    pub fn is_known(&self) -> bool {
        match self {
            ast::Type::Unknown => false,
            ast::Type::Impossible 
            | ast::Type::Unit
            | ast::Type::Int
            | ast::Type::Bool
            | ast::Type::Str => true,
            ast::Type::Generic(_) => todo!(),
            ast::Type::Tuple(elements) => elements.iter().all(Self::is_known),
            ast::Type::List(t) => t.is_known(),
            ast::Type::Dict { keys, values } => keys.is_known() && values.is_known(),
            ast::Type::Callable { generics: _, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument, return_type } => {
                positional_arguments.iter().all(|t| t.is_known())
                && variadic_argument.as_ref().map_or(true, |t| t.is_known())
                && keyword_arguments.iter().all(|kw| kw.arg_type.is_known())
                && keyword_variadic_argument.as_ref().map_or(true, |t| t.is_known())
                && return_type.is_known()
            }
        }
    }

    pub fn validate_call(&self,
        caller_positional_arguments: &Vec<ast::Type>,
        caller_variadic_argument: &Option<ast::Type>,
        caller_keyword_arguments: &Vec<ast::KeywordArgumentType>,
        caller_keyword_variadic_argument: &Option<ast::Type>
    ) -> Option<ast::Type> {
        let return_type = match self {
            ast::Type::Callable { return_type, .. } => *return_type.clone(),
            _ => return None
        };

        if let ast::Type::Generic(_) = return_type {
            todo!()
        }

        let caller_expected_type = ast::Type::Callable { 
            generics: vec![],
            positional_arguments: caller_positional_arguments.clone(),
            variadic_argument: caller_variadic_argument.clone().map(Box::new),
            keyword_arguments: caller_keyword_arguments.clone(),
            keyword_variadic_argument: caller_keyword_variadic_argument.clone().map(Box::new),
            return_type: Box::new(return_type.clone()) // this will just automatically succeed in the covariance check
        };

        if caller_expected_type.subtypes(self) {
            Some(return_type)
        } else {
            None
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

impl ProgramEnv {
    pub fn new() -> Self {
        ProgramEnv { functions: HashMap::new() }
    }
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
                match &arg.arg_type_literal {
                    Some(lit) => {
                        lit.typ.validate_generics(&generics, &arg.loc)?;
                        let ann = lit.typ.get_type();
                        let arg_expr = arg.expr.typecheck(&mut FunctionEnv::new())?;
                        if !arg_expr.typ.subtypes(&ann) {
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::NotExpectedType(arg_expr.expr.clone(), ann),
                                loc: arg.loc.clone(),
                            });
                        }
                        
                        Ok(ast::KeywordArgument {
                            name: arg.name,
                            expr: arg_expr,
                            arg_type_literal: arg.arg_type_literal,
                            loc: arg.loc,
                            typ: ann,
                        })
                    },
                    _ => {
                        let arg_expr = arg.expr.typecheck(&mut FunctionEnv::new())?;
                        let arg_expr_typ = arg_expr.typ.clone();
                        if let ast::Type::Unknown = arg_expr_typ {
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::MissingTypeAnnotation,
                                loc: arg.loc.clone(),
                            })
                        } else {
                                Ok(ast::KeywordArgument {
                                name: arg.name,
                                expr: arg_expr,
                                arg_type_literal: arg.arg_type_literal,
                                loc: arg.loc,
                                typ: arg_expr_typ,
                            })
                        }
                    }
                }
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


        Ok(ast::FunctionPrototype {
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

    pub fn new() -> Self {
        FunctionEnv { program_env: ProgramEnv::new(), return_type: ast::Type::Impossible, variable_types: vec![] }
    }

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

    pub fn get_variable_type(&self, var: &String) -> Option<ast::Type> {
        for mapping in self.variable_types.iter() {
            if let Some(existing_typ) = mapping.get(var) {
                return Some(existing_typ.clone());
            }
        }
        return None;
    }

    pub fn is_top_level_block(&self) -> bool {
        self.variable_types.len() == 1
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

                Ok(ast::LocStmt {
                    stmt: ast::Stmt::Return { expr: expr },
                    loc: self.loc,
                    typ: ast::Type::Impossible
                })
            },
            ast::Stmt::IfElse { cond, if_body, else_body } => {
                let cond = cond.typecheck(env)?;

                if !cond.typ.subtypes(&ast::Type::Bool) {
                    return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotExpectedType(cond.expr, ast::Type::Bool),
                        loc: cond.loc
                    })
                }

                let if_body = if_body.typecheck(env)?;
                let else_body = else_body.typecheck(env)?;

                // for now we just join the return types
                let join = if_body.typ.join(&else_body.typ);

                Ok(ast::LocStmt {
                    stmt: ast::Stmt::IfElse { cond, if_body: Box::new(if_body), else_body: Box::new(else_body) },
                    loc: self.loc,
                    typ: join
                })
            },
            ast::Stmt::While { cond, body } => {
                let cond = cond.typecheck(env)?;

                if !cond.typ.subtypes(&ast::Type::Bool) {
                    return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotExpectedType(cond.expr, ast::Type::Bool),
                        loc: cond.loc
                    })
                }

                let body = body.typecheck(env)?;

                Ok(ast::LocStmt {
                    stmt: ast::Stmt::While { cond: cond, body: Box::new(body) },
                    loc: self.loc,
                    typ: ast::Type::Unit
                })
            },
            ast::Stmt::Block { mut statements } => {
                let mut env = env.clone();

                if let Some(last) = statements.pop() {
                    let mut statements_new: Vec<ast::LocStmt> = vec![];

                    let mut iter = statements.into_iter().peekable();

                    while let Some(stmt) = iter.next() {

                        let stmt: ast::LocStmt = stmt;
                        let stmt = stmt.typecheck(&mut env)?;
                        
                        if let ast::Type::Impossible = stmt.typ {
                            let next_stmt = match iter.peek() {
                                Some(n) => n,
                                _ => &last
                            };
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::Unreachable(),
                                loc: next_stmt.loc.clone()
                            })
                        }

                        statements_new.push(stmt);
                    }

                    let last = last.typecheck(&mut env)?;
                    let last_typ = last.typ.clone();

                    statements_new.push(last);

                    Ok(ast::LocStmt {
                        stmt: ast::Stmt::Block { 
                            statements: statements_new
                        },
                        loc: self.loc,
                        typ: last_typ
                    })
                } else {
                    Ok(ast::LocStmt {
                        stmt: ast::Stmt::Block {statements: vec![]},
                        loc: self.loc,
                        typ: ast::Type::Unit
                    })
                }
            },
            ast::Stmt::SoftBlock { mut statements } => {
                let mut env = env.clone();

                if let Some(last) = statements.pop() {
                    let mut statements_new: Vec<ast::LocStmt> = vec![];

                    let mut iter = statements.into_iter().peekable();

                    while let Some(stmt) = iter.next() {

                        let stmt: ast::LocStmt = stmt;
                        let stmt = stmt.typecheck(&mut env)?;
                        
                        if let ast::Type::Impossible = stmt.typ {
                            let next_stmt = match iter.peek() {
                                Some(n) => n,
                                _ => &last
                            };
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::Unreachable(),
                                loc: next_stmt.loc.clone()
                            })
                        }

                        statements_new.push(stmt);
                    }

                    let last = last.typecheck(&mut env)?;
                    let last_typ = last.typ.clone();

                    statements_new.push(last);

                    Ok(ast::LocStmt {
                        stmt: ast::Stmt::SoftBlock { 
                            statements: statements_new
                        },
                        loc: self.loc,
                        typ: last_typ
                    })
                } else {
                    Ok(ast::LocStmt {
                        stmt: ast::Stmt::SoftBlock {statements: vec![]},
                        loc: self.loc,
                        typ: ast::Type::Unit
                    })
                }
            },
            ast::Stmt::Expression { expr } => {
                let expr = expr.typecheck(env)?;
                let expr_typ = expr.typ.clone();
                Ok(ast::LocStmt {
                    stmt: ast::Stmt::Expression { expr: expr },
                    loc: self.loc,
                    typ: expr_typ
                })
            },
            ast::Stmt::Break => Ok(ast::LocStmt {
                stmt: ast::Stmt::Break,
                loc: self.loc,
                typ: ast::Type::Impossible
            }),
            ast::Stmt::Continue => Ok(ast::LocStmt {
                stmt: ast::Stmt::Continue,
                loc: self.loc,
                typ: ast::Type::Impossible
            }),
        }
    }
}



impl ast::BinOp {

    pub fn signature(&self) -> ast::Type {
        match self {
            ast::BinOp::Eq => todo!(),
            ast::BinOp::Neq => todo!(),
            ast::BinOp::Leq => todo!(),
            ast::BinOp::Geq => todo!(),
            ast::BinOp::Lt => todo!(),
            ast::BinOp::Gt => todo!(),
            ast::BinOp::Add => todo!(),
            ast::BinOp::Sub => todo!(),
            ast::BinOp::Mul => todo!(),
            ast::BinOp::Div => todo!(),
            ast::BinOp::Mod => todo!(),
            ast::BinOp::ShiftLeft => todo!(),
            ast::BinOp::ShiftRightArith => todo!(),
            ast::BinOp::And => todo!(),
            ast::BinOp::Or => todo!(),
            ast::BinOp::In => todo!(),
        }
    }

}

impl ast::UnOp {

    pub fn signature(&self) -> ast::Type {
        match self {
            ast::UnOp::Neg => todo!(),
            ast::UnOp::Not => todo!(),
        }
    }

}


impl ast::LocExpr {

    pub fn typecheck(self, env: &mut FunctionEnv) -> Result<Self, TypecheckingErrorMessage> {
        
        match self.expr {
            ast::Expr::Variable(x) => {
                if let Some(typ) = env.get_variable_type(&x) {
                    Ok(ast::LocExpr {expr: ast::Expr::Variable(x), loc: self.loc, typ})
                } else if let Some(typ) = env.program_env.functions.get(&x) {
                    Ok(ast::LocExpr {expr: ast::Expr::Variable(x), loc: self.loc, typ: typ.clone()})
                } else {
                    Err(TypecheckingErrorMessage {error: TypecheckingError::VariableNotFound(x.clone()), loc: self.loc})
                }
            },
            ast::Expr::DotAccess(_, _) => todo!(),
            ast::Expr::Int(x) => Ok(ast::LocExpr {expr: ast::Expr::Int(x), loc: self.loc, typ: ast::Type::Int}),
            ast::Expr::Bool(b) => Ok(ast::LocExpr {expr: ast::Expr::Bool(b), loc: self.loc, typ: ast::Type::Bool}),
            ast::Expr::Str(s) => Ok(ast::LocExpr {expr: ast::Expr::Str(s), loc: self.loc, typ: ast::Type::Str}),
            ast::Expr::Tuple(elements) => {
                let elements: Vec<_> = elements.into_iter().map(|x| Self::typecheck(x, env)).collect::<Result<_,_>>()?;
                let element_types = elements.iter().map(|x| x.typ.clone()).collect();
                Ok(ast::LocExpr {expr: ast::Expr::Tuple(elements), loc: self.loc, typ: ast::Type::Tuple(element_types)})
            },
            ast::Expr::List(elements) => {
                todo!("How do I deal with empty lists...")
            },
            ast::Expr::Dictionary(elements) => {
                todo!("How do I deal with empty lists...")
            },
            ast::Expr::BinOp { op, left, right } => {
                let signature = op.signature();
                let left = left.typecheck(env)?;
                let right = right.typecheck(env)?;

                match signature.validate_call(&vec![left.typ.clone(), right.typ.clone()], &None, &vec![], &None) {
                    Some(return_type) => {
                        Ok(ast::LocExpr {
                            expr: ast::Expr::BinOp { op, left: Box::new(left), right: Box::new(right) },
                            loc: self.loc,
                            typ: return_type
                        })
                    }
                    _ => return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::ArgumentsDontMatchFunction(signature.clone()),
                        loc: self.loc.clone()
                    })
                }
            },
            ast::Expr::UnOp { op, expr } => {
                let signature = op.signature();
                let expr = expr.typecheck(env)?;

                match signature.validate_call(&vec![expr.typ.clone()], &None, &vec![], &None) {
                    Some(return_type) => {
                        Ok(ast::LocExpr {
                            expr: ast::Expr::UnOp { op, expr: Box::new(expr) },
                            loc: self.loc,
                            typ: return_type
                        })
                    }
                    _ => return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::ArgumentsDontMatchFunction(signature.clone()),
                        loc: self.loc.clone()
                    })
                }
            },
            ast::Expr::FunctionCall { function, positional_arguments, variadic_argument, keyword_arguments, keyword_variadic_argument } => {
                let function = function.typecheck(env)?;

                let positional_arguments: Vec<ast::CallArgument> = positional_arguments.into_iter().map(|x| {
                    let expr = Self::typecheck(*x.expr, env)?;
                    Ok(ast::CallArgument {expr: Box::new(expr), loc: x.loc})
                }).collect::<Result<_, _>>()?;

                let variadic_argument = variadic_argument.map(|x| {
                    let expr = Self::typecheck(*x.expr, env)?;
                    Ok(ast::CallArgument {expr: Box::new(expr), loc: x.loc})
                }).transpose()?;

                let keyword_arguments: Vec<ast::CallKeywordArgument> = keyword_arguments.into_iter().map(|x| {
                    let expr = Self::typecheck(*x.expr, env)?;
                    Ok(ast::CallKeywordArgument {name: x.name, expr: Box::new(expr), loc: x.loc})
                }).collect::<Result<_, _>>()?;

                let keyword_variadic_argument = keyword_variadic_argument.map(|x| {
                    let expr = Self::typecheck(*x.expr, env)?;
                    Ok(ast::CallArgument {expr: Box::new(expr), loc: x.loc})
                }).transpose()?;

                match function.typ.validate_call(
                    &positional_arguments.iter().map(|x| x.expr.typ.clone()).collect(),
                    &variadic_argument.clone().map(|x| x.expr.typ),
                    &keyword_arguments.iter().map(|x| ast::KeywordArgumentType {name: x.name.clone(), arg_type: x.expr.typ.clone()}).collect(),
                    &keyword_variadic_argument.clone().map(|x| x.expr.typ)
                ) {
                    Some(return_type) => {
                        Ok(ast::LocExpr {
                            expr: ast::Expr::FunctionCall { 
                                function: Box::new(function),
                                positional_arguments,
                                variadic_argument,
                                keyword_arguments,
                                keyword_variadic_argument 
                            },
                            loc: self.loc,
                            typ: return_type
                        })
                    }
                    _ => return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::ArgumentsDontMatchFunction(function.typ.clone()),
                        loc: self.loc.clone()
                    })
                }
            },
            ast::Expr::Indexing { indexed, indexer } => {
                let indexed = indexed.typecheck(env)?;

                if let ast::Type::Impossible = indexed.typ {
                    return Ok(ast::LocExpr {
                        expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: indexer },
                        loc: self.loc, typ: ast::Type::Impossible
                    });
                }

                match indexed.typ {
                    ast::Type::Generic(s) => todo!("Deep sigh..."),
                    ast::Type::List(ref element_type) => {
                        let indexer = indexer.typecheck(env)?;

                        if let ast::Type::Impossible = indexer.typ {
                            return Ok(ast::LocExpr {
                                expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                loc: self.loc, typ: ast::Type::Impossible
                            });
                        }

                        if !indexer.typ.subtypes(&ast::Type::Int) {
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::NotExpectedType(indexer.expr.clone(), ast::Type::Int),
                                loc: indexer.loc
                            })
                        }

                        let indexed_typ = indexed.typ.clone();
                        

                        Ok(ast::LocExpr {
                            expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                            loc: self.loc, typ: indexed_typ
                        })
                    },
                    ast::Type::Dict{ref keys, ref values} => {
                        let indexer = indexer.typecheck(env)?;

                        if let ast::Type::Impossible = indexer.typ {
                            return Ok(ast::LocExpr {
                                expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                                loc: self.loc, typ: ast::Type::Impossible
                            });
                        }

                        if !indexer.typ.subtypes(&keys) {
                            return Err(TypecheckingErrorMessage {
                                error: TypecheckingError::NotExpectedType(indexer.expr.clone(), *keys.clone()),
                                loc: indexer.loc
                            })
                        }

                        let indexed_typ = indexed.typ.clone();

                        Ok(ast::LocExpr {
                            expr: ast::Expr::Indexing { indexed: Box::new(indexed), indexer: Box::new(indexer) },
                            loc: self.loc, typ: indexed_typ
                        })
                    },
                    _ => return Err(TypecheckingErrorMessage {
                        error: TypecheckingError::NotAValidLocation(ast::Expr::Indexing { indexed: Box::new(indexed.clone()), indexer: indexer.clone() }),
                        loc: self.loc
                    })
                } 
            },
            ast::Expr::Slice { indexed, indexer_start, indexer_border, indexer_step } => todo!(),
            ast::Expr::Lambda { arguments, expr } => todo!(),
            ast::Expr::Block { statements } => todo!(),
        }
    }

}