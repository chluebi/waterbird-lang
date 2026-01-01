use std::collections::{HashMap, HashSet};

use crate::ast::{self, GenericLiteral};

pub enum TypecheckingError {
    MissingTypeAnnotation,
    UnknownType(String),
    NotExpectedType(ast::Expr, ast::Type),
    NotSubtype(ast::Type, ast::Type)
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
            ast::TypeLiteral::Void => ast::Type::Void,
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
            match f.contract.typecheck() {
                Ok((p, t)) => {function_type_mapping.insert(s.clone(), t); Ok((s, ast::Function {name: f.name, contract: p, body: f.body, loc: f.loc}))},
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

    pub fn typecheck(self) -> Result<(Self, ast::Type), TypecheckingErrorMessage> {
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
                todo!()
            },
            _ => None
        };

        let keyword_arguments: Vec<ast::KeywordArgument> = self.keyword_arguments
            .into_iter()
            .map(|arg| {
                let final_type = if let Some(lit) = &arg.arg_type_literal {
                    lit.typ.validate_generics(&generics, &arg.loc)?;
                    let ann = lit.typ.get_type();
                    if !arg.expr.expr.clone().check(&ann)? {
                        return Err(TypecheckingErrorMessage {
                            error: TypecheckingError::NotExpectedType(arg.expr.expr.clone(), ann),
                            loc: arg.loc.clone(),
                        });
                    }
                    ann
                } else {
                    arg.expr.expr.infer()
?
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
        


        todo!()
    }
}


impl ast::Function {

    pub fn verify(self) -> Result<(), TypecheckingErrorMessage> {
        todo!()
    }

    

    pub fn typecheck(self, env: &ProgramEnv) -> Result<Self, TypecheckingErrorMessage> {
        todo!()
    }

}


impl ast::Expr {
    pub fn check(self, expected_type: &ast::Type) -> Result<bool, TypecheckingErrorMessage> {
        todo!()
    }

    pub fn infer(&self) -> Result<Option<ast::Type>, TypecheckingErrorMessage> {
        todo!()
    }
}