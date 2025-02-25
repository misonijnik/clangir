//===--- LowerModule.h - Abstracts CIR's module lowering --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H

#include "CIRLowerContext.h"
#include "LowerTypes.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include <memory>

namespace cir {

class LowerModule {
  CIRLowerContext context;
  mlir::ModuleOp module;
  const std::unique_ptr<clang::TargetInfo> Target;
  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;
  std::unique_ptr<CIRCXXABI> ABI;

  LowerTypes types;

  mlir::PatternRewriter &rewriter;

public:
  LowerModule(clang::LangOptions opts, mlir::ModuleOp &module,
              std::unique_ptr<clang::TargetInfo> target,
              mlir::PatternRewriter &rewriter);
  ~LowerModule() = default;

  // Trivial getters.
  LowerTypes &getTypes() { return types; }
  CIRLowerContext &getContext() { return context; }
  CIRCXXABI &getCXXABI() const { return *ABI; }
  const clang::TargetInfo &getTarget() const { return *Target; }
  mlir::MLIRContext *getMLIRContext() { return module.getContext(); }
  mlir::ModuleOp &getModule() { return module; }

  const cir::CIRDataLayout &getDataLayout() const {
    return types.getDataLayout();
  }

  const TargetLoweringInfo &getTargetLoweringInfo();

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  const clang::TargetInfo &getTargetInfo() const { return *Target; }

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  clang::TargetCXXABI::Kind getCXXABIKind() const {
    auto kind = getTarget().getCXXABI().getKind();
    cir_cconv_assert(!cir::MissingFeatures::langOpts());
    return kind;
  }

  void
  constructAttributeList(llvm::StringRef Name, const LowerFunctionInfo &FI,
                         FuncOp CalleeInfo, // TODO(cir): Implement CalleeInfo?
                         FuncOp newFn, unsigned &CallingConv,
                         bool AttrOnCallSite, bool IsThunk);

  void setCIRFunctionAttributes(FuncOp GD, const LowerFunctionInfo &Info,
                                FuncOp F, bool IsThunk);

  /// Set function attributes for a function declaration.
  void setFunctionAttributes(FuncOp oldFn, FuncOp newFn,
                             bool IsIncompleteFunction, bool IsThunk);

  // Create a CIR FuncOp with with the given signature.
  FuncOp createCIRFunction(llvm::StringRef MangledName, FuncType Ty, FuncOp D,
                           bool ForVTable, bool DontDefer = false,
                           bool IsThunk = false,
                           llvm::ArrayRef<mlir::Attribute> =
                               {}, // TODO(cir): __attribute__(()) stuff.
                           bool IsForDefinition = false);

  // Rewrite CIR FuncOp to match the target ABI.
  llvm::LogicalResult rewriteFunctionDefinition(FuncOp op);

  // Rewrite CIR CallOp to match the target ABI.
  llvm::LogicalResult rewriteFunctionCall(CallOp callOp, FuncOp funcOp = {});
};

std::unique_ptr<LowerModule> createLowerModule(mlir::ModuleOp module,
                                               mlir::PatternRewriter &rewriter);

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
