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
#include "cir-tac/Serializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;
)";

const char *const serializerDefStart = R"(
void Serializer::serializeOperation(mlir::Operation &inst,
                                    protocir::CIROp *pInst,
                                    protocir::CIRModuleID pModuleID,
                                    TypeCache &typeCache,
                                    OperationCache &opCache,
                                    BlockCache &blockCache,
                                    FunctionCache &functionCache

) {
  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
)";

const char *const serializerDefEnd = R"(
}
)";

const char *const serializerCaseStart = R"(
      .Case<cir::{0}>([instID, pInst, pModuleID, &typeCache, &blockCache, &opCache](cir::{0} op) {{
        protocir::CIR{0} p{0};
        pInst->mutable_base()->set_id(instID);
)";

const char *const serializerCaseOffset = R"(
        {0}
)";

const char *const serializerCaseDefineOptionalOperand = R"(
        auto {0} = op.{1}().getDefiningOp();
        if ({0}) {{
          auto {0}ID = internOperation(opCache, {0});

          protocir::CIROpID p{0}ID;
          p{0}ID.set_id({0}ID);

          *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefineOptionalOperandEnd = R"(
        }
)";

const char *const serializerCaseDefineVariadicOperand = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          auto e{0}ID = internOperation(opCache, e{0}.getDefiningOp());
          e{0}Proto->set_id(e{0}ID);
        }
)";

const char *const serializerCaseDefineVariadicOfVeriadicOperand = R"(
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

const char *const serializerCaseDefineOperand = R"(
        auto {0} = op.{1}().getDefiningOp();
)";

const char *const serializerCaseEnd = R"(
        pInst->mutable_{0}()->CopyFrom(p{1});
      })
)";

const char *const serializerDefaultCase = R"(
      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });
)";

const char *const protoOpMessageStart = R"(
message CIR{0} {{)";

const char *const protoOpMessageField = R"(
  {0} {1} = {2};)";

const char *const protoOpMessageEnd = R"(
}
)";

const std::map<StringRef, StringRef> cppTypeToProto = {
    {"uint64_t", "uint64"},
    {"uint32_t", "uint32"},
    {"::llvm::StringRef", "string"},
    {"::llvm::APInt", "uint64"},
    {"::llvm::APFloat", "double"},
    {"::cir::GlobalDtorAttr", "google.protobuf.Empty"},
    {"::cir::GlobalCtorAttr", "google.protobuf.Empty"},
    {"::llvm::ArrayRef<int32_t>", "repeated uint32"},
    {"::mlir::TypedAttr", "google.protobuf.Any"},
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

const std::set<StringRef> typeBlackList = {
    "::std::optional< ::mlir::ArrayAttr >",
    "::std::optional<::mlir::Attribute>",
    "::std::optional<::cir::DynamicCastInfoAttr>",
    "::std::optional<::cir::ASTVarDeclInterface>",
    "::mlir::ArrayAttr",
    "::cir::CmpThreeWayInfoAttr",
    "::cir::BitfieldInfoAttr",
    "::std::optional<::cir::AddressSpaceAttr>",
    "::std::optional<::cir::ASTCallExprInterface>",
    "::cir::ExtraFuncAttributesAttr"};

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
      // const auto &operandName =
      // llvm::convertToSnakeFromCamelCase(operand.name);
      const auto &operandNameProto =
          llvm::convertToSnakeFromCamelCase(operand.name);
      const auto &operandNameCpp =
          llvm::convertToCamelFromSnakeCase(operand.name);
      if (typeBlackList.count(operandType)) {
        --messageIdx;
      } else {
        if (operand.isOptional()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineOptionalOperand, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
          os << serializerCaseDefineOptionalOperandEnd;
          // const auto &operandTypeOptional =
          //     TypeConstraint(
          //         operand.constraint.getDef().getValueAsOptionalDef("baseType"))
          //         .getCppType();
          // auto it = cppTypeToProto.find(operandTypeOptional);
          // const auto &operandTypeProto =
          //     it != cppTypeToProto.end() ? it->second : operandTypeOptional;
          // os << formatv(protoOpMessageField,
          //               formatv("optional {0}", operandTypeProto),
          //               operandName, std::to_string(messageIdx + 1));
        } else if (operand.isVariadicOfVariadic()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineVariadicOfVeriadicOperand, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
        } else if (operand.isVariadic()) {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineVariadicOperand, operandNameCpp,
                        getterName, op.getCppClassName(), operandNameProto);
          // auto it = cppTypeToProto.find(operandType);
          // const auto &operandTypeProto =
          //     it != cppTypeToProto.end() ? it->second : operandType;
          // os << formatv(protoOpMessageField,
          //               formatv("repeated {0}", operandTypeProto),
          //               operandName, std::to_string(messageIdx + 1));
        } else {
          std::string getterName = op.getGetterName(operand.name);
          os << formatv(serializerCaseDefineOperand, operandNameCpp,
                        getterName);
          // auto it = cppTypeToProto.find(operandType);
          // const auto &operandTypeProto =
          //     it != cppTypeToProto.end() ? it->second : operandType;
          // os << formatv(protoOpMessageField, operandTypeProto, operandName,
          //               std::to_string(messageIdx + 1));
        }
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
