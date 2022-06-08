// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cnpy.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>

/// @brief Interface to work with files like input and output
class BaseFile {
public:
    virtual void load_file(const char* fileName,
                           uint32_t arrayIndex,
                           std::string& ptrName,
                           std::vector<uint8_t>& memory,
                           uint32_t* ptrNumRows,
                           uint32_t* ptrNumColumns,
                           uint32_t* ptrNumBytesPerElement) const = 0;

    virtual void save_file(const char* fileName,
                           bool shouldAppend,
                           std::string name,
                           void* ptrMemory,
                           uint32_t numRows,
                           uint32_t numColumns) const = 0;

    virtual void get_file_info(const char* fileName,
                               uint32_t numArrayToFindSize,
                               uint32_t* ptrNumArrays,
                               uint32_t* ptrNumMemoryBytes) const = 0;
};

/// @brief Responsible to work with .ark files
class ArkFile : public BaseFile {
public:
    /**
     * @brief Get info from Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param numArrayToFindSize number speech feature vectors in the file
     * @param ptrNumArrays pointer to specific number array
     * @param ptrNumMemoryBytes pointer to specific number of memory bytes
     * @return none.
     */
    void get_file_info(const char* fileName,
                       uint32_t numArrayToFindSize,
                       uint32_t* ptrNumArrays,
                       uint32_t* ptrNumMemoryBytes) const override;

    /**
     * @brief Load Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param arrayIndex number speech feature vector in the file
     * @param ptrName reference to variable length name
     * @param memory reference to speech feature vector to save
     * @param ptrNumRows pointer to number of rows to read
     * @param ptrNumColumns pointer to number of columns to read
     * @param ptrNumBytesPerElement pointer to number bytes per element (size of float by default)
     * @return none.
     */
    void load_file(const char* fileName,
                   uint32_t arrayIndex,
                   std::string& ptrName,
                   std::vector<uint8_t>& memory,
                   uint32_t* ptrNumRows,
                   uint32_t* ptrNumColumns,
                   uint32_t* ptrNumBytesPerElement) const override;

    /**
     * @brief Save Kaldi ARK speech feature vector file
     * @param fileName .ark file name
     * @param shouldAppend bool flag to rewrite or add to the end of file
     * @param name reference to variable length name
     * @param ptrMemory pointer to speech feature vector to save
     * @param numRows number of rows
     * @param numColumns number of columns
     * @return none.
     */
    void save_file(const char* fileName,
                   bool shouldAppend,
                   std::string name,
                   void* ptrMemory,
                   uint32_t numRows,
                   uint32_t numColumns) const override;
};

/// @brief Responsible to work with .npz files
class NumpyFile : public BaseFile {
public:
    /**
     * @brief Get info from Numpy* uncompressed NPZ speech feature vector file
     * @param fileName .npz file name
     * @param numArrayToFindSize number speech feature vectors in the file
     * @param ptrNumArrays pointer to specific number array
     * @param ptrNumMemoryBytes pointer to specific number of memory bytes
     * @return none.
     */
    void get_file_info(const char* fileName,
                       uint32_t numArrayToFindSize,
                       uint32_t* ptrNumArrays,
                       uint32_t* ptrNumMemoryBytes) const override;

    /**
     * @brief Load Numpy* uncompressed NPZ speech feature vector file
     * @param fileName .npz file name
     * @param arrayIndex number speech feature vector in the file
     * @param ptrName reference to variable length name
     * @param memory reference to speech feature vector to save
     * @param ptrNumRows pointer to number of rows to read
     * @param ptrNumColumns pointer to number of columns to read
     * @param ptrNumBytesPerElement pointer to number bytes per element (size of float by default)
     * @return none.
     */
    void load_file(const char* fileName,
                   uint32_t arrayIndex,
                   std::string& ptrName,
                   std::vector<uint8_t>& memory,
                   uint32_t* ptrNumRows,
                   uint32_t* ptrNumColumns,
                   uint32_t* ptrNumBytesPerElement) const override;

    /**
     * @brief Save Numpy* uncompressed NPZ speech feature vector file
     * @param fileName .npz file name
     * @param shouldAppend bool flag to rewrite or add to the end of file
     * @param name reference to variable length name
     * @param ptrMemory pointer to speech feature vector to save
     * @param numRows number of rows
     * @param numColumns number of columns
     * @return none.
     */
    void save_file(const char* fileName,
                   bool shouldAppend,
                   std::string name,
                   void* ptrMemory,
                   uint32_t numRows,
                   uint32_t numColumns) const override;
};

/**
 * @brief Facade class allowing to support multiple implementation of BaseFile. Specific implementation is chosen
 *     based on the file extension
 */
class FileHandler : public BaseFile {
public:
    /**
     * Construct {FileHandler} object instance.
     */
    FileHandler();

protected:
    /**
     * @see BaseFile.load_file()
     * @throw logic_error in case unssported file format is used.
     */
    void load_file(const char* fileName,
                   uint32_t arrayIndex,
                   std::string& ptrName,
                   std::vector<uint8_t>& memory,
                   uint32_t* ptrNumRows,
                   uint32_t* ptrNumColumns,
                   uint32_t* ptrNumBytesPerElement) const override;

    /**
     * @see BaseFile.save_file()
     * @throw logic_error in case unssported file format is used.
     */
    void save_file(const char* fileName,
                   bool shouldAppend,
                   std::string name,
                   void* ptrMemory,
                   uint32_t numRows,
                   uint32_t numColumns) const override;

    /**
     * @see BaseFile.get_file_info()
     * @throw logic_error in case unssported file format is used.
     */
    virtual void get_file_info(const char* fileName,
                               uint32_t numArrayToFindSize,
                               uint32_t* ptrNumArrays,
                               uint32_t* ptrNumMemoryBytes) const override;

private:
    BaseFile& get_file_format_hanlder(const char* fileName) const;

    static const std::string kArkFileExt;
    static const std::string kNumpyFileExt;

    std::unordered_map<std::string, std::unique_ptr<BaseFile>> supproted_file_formats_;
};