use std::collections::HashMap;

use crate::ast;


enum TypecheckingError {

}


impl ast::Program {

    pub fn verify(self) -> Result<(), TypecheckingError> {
        todo!()
    }

    pub fn typecheck(self) -> Result<Self, TypecheckingError> {
        let res: Result<HashMap<String, ast::Function>, TypecheckingError> = self.functions.into_iter().map(|(s,f)| {
            match f.typecheck() {
                Ok(f) => Ok((s, f)),
                Err(e) => Err(e)
            }
        }).collect();
        Ok(ast::Program {functions: res?})
    }

}


impl ast::Function {

    pub fn verify(self) -> Result<(), TypecheckingError> {
        todo!()
    }

    pub fn typecheck(self) -> Result<Self, TypecheckingError> {
        todo!()
    }

}