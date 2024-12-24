//===- ProtoOpSerializerGen.cpp - Proto op serializer generator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ProtoOpDefinitionsGen uses the description of operations to generate Proto
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "OpGenHelpers.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <map>
#include <set>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

const char *const serializerFileHeader = R"(
#include "cir-tac/EnumsSerializer.h"
#include "cir-tac/Serializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;
)";

const char *const serializerDefStart = R"(
protocir::CIROp Serializer::serializeOperation(mlir::Operation &inst,
                                               protocir::CIRModuleID pModuleID,
                                               TypeCache &typeCache,
                                               OperationCache &opCache,
                                               BlockCache &blockCache) {
  protocir::CIROp pInst;

  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
)";

const char *const serializerDefEnd = R"(
  return pInst;
}
)";

const char *const serializerCaseStart = R"(
      .Case<cir::{0}>([instID, &pInst, pModuleID, &typeCache, &blockCache, &opCache](cir::{0} op) {{
        protocir::CIR{0} p{0};
        pInst.mutable_base()->set_id(instID);

        auto resultTypes = op.getOperation()->getResultTypes();
        for (const auto &resultType : resultTypes) {{
          auto resultTypeID = internType(typeCache, resultType);
          pInst.add_result_types()->set_id(resultTypeID);
        }
)";

const char *const serializerCaseDefineOptionalOperation = R"(
        auto {0}Raw = op.{1}();
        if ({0}Raw) {{
          auto {0} = {0}Raw.getDefiningOp();
          auto {0}ID = internOperation(opCache, {0});

          protocir::CIROpID p{0}ID;
          p{0}ID.set_id({0}ID);

          *p{2}.mutable_{3}() = p{0}ID;
        }
)";

const char *const serializerCaseDefineVariadicOperation = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          auto e{0}ID = internOperation(opCache, e{0}.getDefiningOp());
          e{0}Proto->set_id(e{0}ID);
        }
)";

const char *const serializerCaseDefineVariadicOfVariadicOperation = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          for (auto ee{0} : e{0}) {{
            auto ee{0}Proto = e{0}Proto->add_range();
            auto ee{0}ID = internOperation(opCache, ee{0}.getDefiningOp());
            ee{0}Proto->set_id(ee{0}ID);
          }
        }
)";

const char *const serializerCaseDefineOperation = R"(
        auto {0} = op.{1}().getDefiningOp();
        auto {0}ID = internOperation(opCache, {0});

        protocir::CIROpID p{0}ID;
        p{0}ID.set_id({0}ID);

        *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefinePrimitive = R"(
        auto {0} = op.{1}();
        p{2}.set_{3}({0});
)";

const char *const serializerCaseDefineType = R"(
        auto {0} = op.{1}();
        auto {0}ID = internType(typeCache, {0});

        protocir::CIRTypeID p{0}ID;
        p{0}ID.set_id({0}ID);

        *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefineOptional = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          *p{2}.mutable_{3}() = {0};
        }
)";

const char *const serializerCaseDefineOptionalPrimitive = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          p{2}.set_{3}({0});
        }
)";

const char *const serializerCaseDefineOptionalCtorDtor = R"(
        auto {0}Optional = op.{1}();
        p{2}.set_{3}({0}Optional.has_value());
)";

const char *const serializerCaseDefine = R"(
        auto {0} = op.{1}();
        *p{2}.mutable_{3}() = {0};
)";

const char *const serializerCaseDefineAPInt = R"(
        llvm::SmallVector<char> {0}Str;
        auto {0} = op.{1}();
        {0}.toString({0}Str, 10, false); 
        llvm::StringRef {0}StrRef({0}Str.data(), {0}Str.size());
        *p{2}.mutable_{3}() = {0}StrRef;

)";

const char *const serializerCaseDefineOptionalAPFloat = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          llvm::SmallVector<char> {0}Str;
          {0}.toString({0}Str); 
          llvm::StringRef {0}StrRef({0}Str.data(), {0}Str.size());
          *p{2}.mutable_{3}() = {0}StrRef;
        }
)";

const char *const serializerCaseDefineTypedAttr = R"(
        std::string {0}Str;
        llvm::raw_string_ostream {0}RawStream({0}Str);
        op.{1}().print({0}RawStream);
        *p{2}.mutable_{3}() = {0}Str;
)";

const char *const serializerCaseDefineVariadicPrimitive = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          p{2}.add_{3}(e{0});
        }
)";

const char *const serializerCaseDefineEnum = R"(
        auto {0} = op.{1}();
        auto p{0} = EnumSerializer::serialize{4}({0});
        p{2}.set_{3}(p{0});
)";

const char *const serializerCaseDefineEnumAttr = R"(
        auto {0} = op.{1}().getValue();
        auto p{0} = EnumSerializer::serialize{4}({0});
        p{2}.set_{3}(p{0});
)";

const char *const serializerCaseDefineOptionalEnum = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          auto p{0} = EnumSerializer::serialize{4}({0});
          p{2}.set_{3}(p{0});
        }
)";

const char *const serializerCaseDefineOptionalAttribute = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();

          std::string {0}Str;
          llvm::raw_string_ostream {0}RawStream({0}Str);
          {0}.print({0}RawStream);
          *p{2}.mutable_{3}() = {0}Str;
        }
)";

const char *const serializerCaseDefineSuccessor = R"(
        auto {0} = op.{1}();
        auto {0}ID = internBlock(blockCache, {0});
        protocir::CIRBlockID p{0}ID;
        p{0}ID.set_id({0}ID);

        *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefineVariadicSuccessor = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          auto e{0}ID = internBlock(blockCache, e{0});
          protocir::CIRBlockID pe{0}ID;
          pe{0}ID.set_id(e{0}ID);
        }
)";

const char *const serializerCaseEnd = R"(
        pInst.mutable_{0}()->CopyFrom(p{1});
      })
)";

const char *const serializerDefaultCase = R"(
      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });
)";

const std::map<StringRef, StringRef> cppAttrTypeToProto = {
    {"uint64_t", "uint64"},
    {"uint32_t", "uint32"},
    {"::llvm::StringRef", "string"},
    {"::llvm::APInt", "string"},
    {"::llvm::APFloat", "string"},
    {"::cir::GlobalDtorAttr", "bool"},
    {"::cir::GlobalCtorAttr", "bool"},
    {"::llvm::ArrayRef<int32_t>", "repeated uint32"},
    {"::mlir::Attribute", "string"},
    {"::mlir::TypedAttr", "string"},
    {"::cir::VisibilityAttr", "CIRVisibilityKind"},
    {"::cir::FuncType", "CIROpID"},
    {"::mlir::Type", "CIRTypeID"},
    {"::cir::PointerType", "CIROpID"},
    {"::cir::IntType", "CIROpID"},
    {"::cir::MethodType", "CIROpID"},
    {"::cir::DataMemberType", "CIROpID"},
    {"::cir::ComplexType", "CIROpID"},
    {"::cir::VectorType", "CIROpID"},
    {"::cir::BoolType", "CIROpID"}};

const std::map<StringRef, StringRef> cppOperandTypeToProto = {
    {"uint64_t", "uint64"},
    {"uint32_t", "uint32"},
    {"::llvm::StringRef", "string"},
    {"::llvm::APInt", "string"},
    {"::llvm::APFloat", "string"},
    {"::llvm::ArrayRef<int32_t>", "repeated uint32"},
    {"::mlir::TypedAttr", "string"},
    {"::cir::VisibilityAttr", "CIRVisibilityKind"},
    {"::cir::FuncType", "CIROpID"},
    {"::mlir::Type", "CIROpID"},
    {"::cir::PointerType", "CIROpID"},
    {"::cir::IntType", "CIROpID"},
    {"::cir::MethodType", "CIROpID"},
    {"::cir::DataMemberType", "CIROpID"},
    {"::cir::ComplexType", "CIROpID"},
    {"::cir::VectorType", "CIROpID"},
    {"::cir::BoolType", "CIROpID"}};

const std::set<StringRef> typesBlackList = {
    "::std::optional< ::mlir::ArrayAttr >",
    "::std::optional<::cir::DynamicCastInfoAttr>",
    "::std::optional<::cir::ASTVarDeclInterface>",
    "::mlir::ArrayAttr",
    "::cir::CmpThreeWayInfoAttr",
    "::cir::BitfieldInfoAttr",
    "::std::optional<::cir::AddressSpaceAttr>",
    "::std::optional<::cir::ASTCallExprInterface>",
    "::cir::ExtraFuncAttributesAttr"};

static void emitOptionalAttributeSerializer(
    Operator &op, const llvm::StringRef &attrName,
    const llvm::StringRef &attrNameCpp, const llvm::StringRef &attrNameProto,
    const llvm::StringRef &attrType, const llvm::StringRef &attrTypeProto,
    raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  if (attrType == "bool") {
    os << formatv(serializerCaseDefinePrimitive, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::APFloat") {
    os << formatv(serializerCaseDefineOptionalAPFloat, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "uint32_t" || attrType == "uint64_t") {
    os << formatv(serializerCaseDefineOptionalPrimitive, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else if (attrType == "::cir::GlobalCtorAttr" ||
             attrType == "::cir::GlobalDtorAttr") {
    os << formatv(serializerCaseDefineOptionalCtorDtor, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::Attribute") {
    os << formatv(serializerCaseDefineOptionalAttribute, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else {
    os << formatv(serializerCaseDefineOptional, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  }
}

static void emitEnumAttributeSerializer(Operator &op,
                                        const llvm::StringRef &attrName,
                                        const llvm::StringRef &enumName,
                                        const llvm::StringRef &attrNameCpp,
                                        const llvm::StringRef &attrNameProto,
                                        raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  os << formatv(serializerCaseDefineEnum, attrNameCpp, getterName,
                op.getCppClassName(), attrNameProto, enumName);
}

static void emitAttributeSerializer(Operator &op,
                                    const llvm::StringRef &attrName,
                                    const llvm::StringRef &attrNameCpp,
                                    const llvm::StringRef &attrNameProto,
                                    const llvm::StringRef &attrType,
                                    const llvm::StringRef &attrTypeProto,
                                    raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  if (attrType == "bool" || attrType == "uint32_t" || attrType == "uint64_t") {
    os << formatv(serializerCaseDefinePrimitive, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::TypedAttr") {
    os << formatv(serializerCaseDefineTypedAttr, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::APInt") {
    os << formatv(serializerCaseDefineAPInt, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::ArrayRef<int32_t>") {
    os << formatv(serializerCaseDefineVariadicPrimitive, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::Type" || attrType == "::cir::FuncType") {
    os << formatv(serializerCaseDefineType, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::cir::VisibilityAttr") {
    std::string getterName = op.getGetterName(attrName);
    os << formatv(serializerCaseDefineEnumAttr, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto, "VisibilityKind");
  } else {
    os << formatv(serializerCaseDefine, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  }
}

static void emitOptionalEnumAttributeSerializer(
    Operator &op, const llvm::StringRef &attrName,
    const llvm::StringRef &enumName, const llvm::StringRef &attrNameCpp,
    const llvm::StringRef &attrNameProto, raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  os << formatv(serializerCaseDefineOptionalEnum, attrNameCpp, getterName,
                op.getCppClassName(), attrNameProto, enumName);
}

static bool emitOpProtoSerializer(const RecordKeeper &records,
                                  raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit. */\n";
  os << serializerFileHeader;
  os << serializerDefStart;

  std::vector<const Record *> defs = getRequestedOpDefinitions(records);
  for (auto *def : defs) {
    Operator op(*def);
    const int numOperands = op.getNumOperands();
    os << formatv(serializerCaseStart, op.getCppClassName());
    int messageIdx = 0;
    for (int i = 0; i != numOperands; ++i, ++messageIdx) {
      const auto &operand = op.getOperand(i);
      const auto &operandType = operand.constraint.getCppType();
      if (operand.name.empty())
        continue;
      const auto &operandNameProto =
          llvm::convertToSnakeFromCamelCase(operand.name);
      const auto &operandNameCpp =
          llvm::convertToCamelFromSnakeCase(operand.name);
      if (typesBlackList.count(operandType)) {
        --messageIdx;
      } else {
        if (operand.isOptional()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineOptionalOperation, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
        } else if (operand.isVariadicOfVariadic()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineVariadicOfVariadicOperation,
                        operandNameCpp, getterName, op.getCppClassName(),
                        operandNameProto);
        } else if (operand.isVariadic()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineVariadicOperation, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
        } else {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineOperation, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
        }
      }
    }
    os << "\n";
    const int numAttributes = op.getNumNativeAttributes();
    for (int i = 0; i != numAttributes; ++i, ++messageIdx) {
      const auto &attr = op.getAttribute(i).attr;
      const auto &attrName = op.getAttribute(i).name;
      if (attrName.empty())
        continue;
      const auto &attrNameProto = llvm::convertToSnakeFromCamelCase(attrName);
      const auto &attrNameCpp = llvm::convertToCamelFromSnakeCase(attrName);
      const auto &attrType = attr.getReturnType();
      if (typesBlackList.count(attrType)) {
        --messageIdx;
      } else if (attr.isEnumAttr()) {
        EnumAttr enumAttr(attr.getDef());
        StringRef enumName = enumAttr.getEnumClassName();
        emitEnumAttributeSerializer(op, attrName, enumName, attrNameCpp,
                                    attrNameProto, os);
      } else if (attr.isOptional() && attr.getBaseAttr().isEnumAttr()) {
        EnumAttr enumAttr(attr.getBaseAttr().getDef());
        StringRef enumName = enumAttr.getEnumClassName();
        emitOptionalEnumAttributeSerializer(op, attrName, enumName, attrNameCpp,
                                            attrNameProto, os);
      } else if (attr.hasDefaultValue() && attr.getBaseAttr().isEnumAttr()) {
        EnumAttr enumAttr(attr.getBaseAttr().getDef());
        StringRef enumName = enumAttr.getEnumClassName();
        emitEnumAttributeSerializer(op, attrName, enumName, attrNameCpp,
                                    attrNameProto, os);
      } else if (attr.isOptional()) {
        Attribute baseAttr(&attr.getBaseAttr().getDef());
        const auto &baseAttrType = baseAttr.getReturnType();
        auto it = cppAttrTypeToProto.find(baseAttrType);
        auto &attrTypeProto =
            it != cppAttrTypeToProto.end() ? it->second : baseAttrType;

        emitOptionalAttributeSerializer(op, attrName, attrNameCpp,
                                        attrNameProto, baseAttrType,
                                        attrTypeProto, os);
      } else {
        auto it = cppAttrTypeToProto.find(attrType);
        auto &attrTypeProto =
            it != cppAttrTypeToProto.end() ? it->second : attrType;

        emitAttributeSerializer(op, attrName, attrNameCpp, attrNameProto,
                                attrType, attrTypeProto, os);
      }
    }
    os << "\n";
    unsigned numSuccessors = op.getNumSuccessors();
    for (unsigned i = 0; i != numSuccessors; ++i, ++messageIdx) {
      const NamedSuccessor &successor = op.getSuccessor(i);
      if (successor.name.empty())
        continue;
      const auto &successorNameProto =
          llvm::convertToSnakeFromCamelCase(successor.name);
      const auto &successorNameCpp =
          llvm::convertToCamelFromSnakeCase(successor.name);
      std::string getterName = op.getGetterName(successor.name);
      if (successor.isVariadic()) {
        os << formatv(serializerCaseDefineVariadicSuccessor, successorNameCpp,
                      getterName, op.getCppClassName(), successorNameProto);
      } else {
        os << formatv(serializerCaseDefineSuccessor, successorNameCpp,
                      getterName, op.getCppClassName(), successorNameProto);
      }
    }
    os << "\n";
    os << formatv(serializerCaseEnd,
                  llvm::convertToSnakeFromCamelCase(op.getCppClassName()),
                  op.getCppClassName());
  }
  os << serializerDefaultCase;

  os << serializerDefEnd;
  return false;
}

static mlir::GenRegistration
    genOpSerializerProto("gen-op-ser-proto",
                         "Generate serializer to op Proto definitions",
                         &emitOpProtoSerializer);
