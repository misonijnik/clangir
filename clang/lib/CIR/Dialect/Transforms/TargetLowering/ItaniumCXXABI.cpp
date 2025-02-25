//===------- ItaniumCXXABI.cpp - Emit CIR code Itanium-specific code  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides CIR lowering logic targeting the Itanium C++ ABI. The class in
// this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
// This file partially mimics clang/lib/CodeGen/ItaniumCXXABI.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "../LoweringPrepareCXXABI.h"
#include "CIRCXXABI.h"
#include "LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class ItaniumCXXABI : public CIRCXXABI {

protected:
  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;
  bool Use32BitVTableOffsetABI;

public:
  ItaniumCXXABI(LowerModule &LM, bool UseARMMethodPtrABI = false,
                bool UseARMGuardVarABI = false)
      : CIRCXXABI(LM), UseARMMethodPtrABI(UseARMMethodPtrABI),
        UseARMGuardVarABI(UseARMGuardVarABI), Use32BitVTableOffsetABI(false) {}

  bool classifyReturnType(LowerFunctionInfo &FI) const override;

  // FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  RecordArgABI getRecordArgABI(const StructType RD) const override {
    cir_cconv_assert(!cir::MissingFeatures::recordDeclIsCXXDecl());
    // If C++ prohibits us from making a copy, pass by address.
    cir_cconv_assert(!cir::MissingFeatures::recordDeclCanPassInRegisters());
    return RAA_Default;
  }

  mlir::Type
  lowerDataMemberType(cir::DataMemberType type,
                      const mlir::TypeConverter &typeConverter) const override;

  mlir::TypedAttr lowerDataMemberConstant(
      cir::DataMemberAttr attr, const mlir::DataLayout &layout,
      const mlir::TypeConverter &typeConverter) const override;

  mlir::Operation *
  lowerGetRuntimeMember(cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
                        mlir::Value loweredAddr, mlir::Value loweredMember,
                        mlir::OpBuilder &builder) const override;
};

} // namespace

bool ItaniumCXXABI::classifyReturnType(LowerFunctionInfo &FI) const {
  const StructType RD = mlir::dyn_cast<StructType>(FI.getReturnType());
  if (!RD)
    return false;

  // If C++ prohibits us from making a copy, return by address.
  if (cir::MissingFeatures::recordDeclCanPassInRegisters())
    cir_cconv_unreachable("NYI");

  return false;
}

mlir::Type ItaniumCXXABI::lowerDataMemberType(
    cir::DataMemberType type, const mlir::TypeConverter &typeConverter) const {
  // Itanium C++ ABI 2.3:
  //   A pointer to data member is an offset from the base address of
  //   the class object containing it, represented as a ptrdiff_t
  const clang::TargetInfo &target = LM.getTarget();
  clang::TargetInfo::IntType ptrdiffTy =
      target.getPtrDiffType(clang::LangAS::Default);
  return cir::IntType::get(type.getContext(), target.getTypeWidth(ptrdiffTy),
                           target.isTypeSigned(ptrdiffTy));
}

mlir::TypedAttr ItaniumCXXABI::lowerDataMemberConstant(
    cir::DataMemberAttr attr, const mlir::DataLayout &layout,
    const mlir::TypeConverter &typeConverter) const {
  uint64_t memberOffset;
  if (attr.isNullPtr()) {
    // Itanium C++ ABI 2.3:
    //   A NULL pointer is represented as -1.
    memberOffset = -1ull;
  } else {
    // Itanium C++ ABI 2.3:
    //   A pointer to data member is an offset from the base address of
    //   the class object containing it, represented as a ptrdiff_t
    auto memberIndex = attr.getMemberIndex().value();
    memberOffset =
        attr.getType().getClsTy().getElementOffset(layout, memberIndex);
  }

  mlir::Type abiTy = lowerDataMemberType(attr.getType(), typeConverter);
  return cir::IntAttr::get(abiTy, memberOffset);
}

mlir::Operation *ItaniumCXXABI::lowerGetRuntimeMember(
    cir::GetRuntimeMemberOp op, mlir::Type loweredResultTy,
    mlir::Value loweredAddr, mlir::Value loweredMember,
    mlir::OpBuilder &builder) const {
  auto byteTy = IntType::get(op.getContext(), 8, true);
  auto bytePtrTy = PointerType::get(
      byteTy, mlir::cast<PointerType>(op.getAddr().getType()).getAddrSpace());
  auto objectBytesPtr = builder.create<CastOp>(op.getLoc(), bytePtrTy,
                                               CastKind::bitcast, op.getAddr());
  auto memberBytesPtr = builder.create<PtrStrideOp>(
      op.getLoc(), bytePtrTy, objectBytesPtr, loweredMember);
  return builder.create<CastOp>(op.getLoc(), op.getType(), CastKind::bitcast,
                                memberBytesPtr);
}

CIRCXXABI *CreateItaniumCXXABI(LowerModule &LM) {
  switch (LM.getCXXABIKind()) {
  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::AppleARM64:
    // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
    // from ARMCXXABI. We'll have to follow suit.
    cir_cconv_assert(!cir::MissingFeatures::appleArm64CXXABI());
    return new ItaniumCXXABI(LM, /*UseARMMethodPtrABI=*/true,
                             /*UseARMGuardVarABI=*/true);

  case clang::TargetCXXABI::GenericItanium:
    return new ItaniumCXXABI(LM);

  case clang::TargetCXXABI::Microsoft:
    cir_cconv_unreachable("Microsoft ABI is not Itanium-based");
  default:
    cir_cconv_unreachable("NYI");
  }

  cir_cconv_unreachable("bad ABI kind");
}

} // namespace cir

// FIXME(cir): Merge this into the CIRCXXABI class above.
class LoweringPrepareItaniumCXXABI : public cir::LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                               clang::ASTContext &astCtx,
                               cir::DynamicCastOp op) override;
  mlir::Value lowerVAArg(cir::CIRBaseBuilderTy &builder, cir::VAArgOp op,
                         const cir::CIRDataLayout &datalayout) override;
};
