/*!
*  Copyright (c) 2016 by Contributors
* \file ndarray.h
* \brief definition of ndarray
* \author Chuntao Hong, Zhang Chen
*/

#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_NDARRAY_H_
#define CPP_PACKAGE_INCLUDE_MXNET_CPP_NDARRAY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdint.h>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"

namespace mxnet {
namespace cpp {

enum DeviceType {
  kCPU = 1,
  kGPU = 2,
  kCPUPinned = 3
};

enum DataTypeFlag {
  kFloat32,
  kFloat64,
  kFloat16,
  kUint8,
  kInt32
};

/*!
* \brief Context interface
*/
class Context {
 public:
  /*!
  * \brief Context constructor
  * \param type type of the device
  * \param id id of the device
  */
  Context(const Context& _context) : type_(_context.type_), id_(_context.id_) {}
  /*!
  * \brief Context constructor
  * \param type type of the device
  * \param id id of the device
  */
  Context(const DeviceType &type, int id) : type_(type), id_(id) {}
  /*!
  * \return the type of the device
  */
  DeviceType GetDeviceType() const { return type_; }
  /*!
  * \return the id of the device
  */
  int GetDeviceId() const { return id_; }

  /*!
   * \brief Return a GPU context
   * \param device_id id of the device
   * \return the corresponding GPU context
   */
  static Context gpu(int device_id = 0) {
    return Context(DeviceType::kGPU, device_id);
  }

  /*!
   * \brief Return a CPU context
   * \param device_id id of the device. this is not needed by CPU
   * \return the corresponding CPU context
   */
  static Context cpu(int device_id = 0) {
    return Context(DeviceType::kCPU, device_id);
  }

 private:
  DeviceType type_;
  int id_;
};

/*!
* \brief struct to store NDArrayHandle
*/
struct NDBlob {
 public:
  /*!
  * \brief default constructor
  */
  NDBlob() : handle_(nullptr) {}
  /*!
  * \brief construct with a NDArrayHandle
  * \param handle NDArrayHandle to store
  */
  explicit NDBlob(NDArrayHandle handle) : handle_(handle) {}
  /*!
  * \brief destructor, free the NDArrayHandle
  */
  ~NDBlob() { MXNDArrayFree(handle_); }
  /*!
  * \brief the NDArrayHandle
  */
  NDArrayHandle handle_;

 private:
  NDBlob(const NDBlob &);
  NDBlob &operator=(const NDBlob &);
};

/*!
* \brief NDArray interface
*/
class NDArray {
 public:
  template <typename T>
  static DataTypeFlag getType();

  template <>
  static DataTypeFlag getType<float>() {
    return kFloat32;
  }

  template <>
  static DataTypeFlag getType<double>() {
    return kFloat64;
  }

  template <>
  static DataTypeFlag getType<uint8_t>() {
    return kUint8;
  }

  template <>
  static DataTypeFlag getType<int32_t>() {
    return kInt32;
  }

  /*!
  * \brief construct with a none handle
  */
  NDArray();
  /*!
  * \brief construct with a NDArrayHandle
  */
  explicit NDArray(const NDArrayHandle &handle);
  /*!
  * \brief construct a new dynamic NDArray
  * \param shape the shape of array
  * \param constext context of NDArray
  * \param delay_alloc whether delay the allocation
  * \param dtype data type of NDArray
  */
  NDArray(const std::vector<mx_uint> &shape, const Context &context,
          bool delay_alloc = true, int dtype = kFloat32);
  /*!
  * \brief construct a new dynamic NDArray
  * \param shape the shape of array
  * \param constext context of NDArray
  * \param delay_alloc whether delay the allocation
  * \param dtype data type of NDArray
  */
  NDArray(const Shape &shape, const Context &context,
          bool delay_alloc = true, int dtype = kFloat32);
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  */
  template<typename T>
  NDArray(const T *data, const Shape &shape, const Context &context);
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  * \param dtype data type of NDArray
  */
  NDArray(const void *data, const Shape &shape, const Context &context, int dtype = kFloat32);
  /*!
  * \brief construct a new dynamic NDArray
  * \param data the data to create NDArray from
  * \param shape the shape of array
  * \param constext context of NDArray
  */
  template<typename T>
  NDArray(const std::vector<T> &data, const Shape &shape,
          const Context &context);
  NDArray operator+(mx_float scalar);
  NDArray operator-(mx_float scalar);
  NDArray operator*(mx_float scalar);
  NDArray operator/(mx_float scalar);
  NDArray operator+(const NDArray &);
  NDArray operator-(const NDArray &);
  NDArray operator*(const NDArray &);
  NDArray operator/(const NDArray &);
  /*!
  * \brief set all the elements in ndarray to be scalar
  * \param scalar the scalar to set
  * \return reference of self
  */
  NDArray &operator=(mx_float scalar);
  /*!
  * \brief elementwise add to current space
  *  this mutate the current NDArray
  * \param scalar the data to add
  * \return reference of self
  */
  NDArray &operator+=(mx_float scalar);
  /*!
  * \brief elementwise subtract from current ndarray
  * this mutate the current NDArray
  * \param scalar the data to substract
  * \return reference of self
  */
  NDArray &operator-=(mx_float scalar);
  /*!
  * \brief elementwise multiplication to current ndarray
  *  this mutate the current NDArray
  * \param scalar the data to substract
  * \return reference of self
  */
  NDArray &operator*=(mx_float scalar);
  /*!
  * \brief elementwise division from current ndarray
  *  this mutate the current NDArray
  * \param scalar the data to substract
  * \return reference of self
  */
  NDArray &operator/=(mx_float scalar);
  /*!
  * \brief elementwise add to current space
  *  this mutate the current NDArray
  * \param src the data to add
  * \return reference of self
  */
  NDArray &operator+=(const NDArray &src);
  /*!
  * \brief elementwise subtract from current ndarray
  * this mutate the current NDArray
  * \param src the data to substract
  * \return reference of self
  */
  NDArray &operator-=(const NDArray &src);
  /*!
  * \brief elementwise multiplication to current ndarray
  *  this mutate the current NDArray
  * \param src the data to substract
  * \return reference of self
  */
  NDArray &operator*=(const NDArray &src);
  /*!
  * \brief elementwise division from current ndarray
  *  this mutate the current NDArray
  * \param src the data to substract
  * \return reference of self
  */
  NDArray &operator/=(const NDArray &src);
  NDArray ArgmaxChannel();
  /*!
  * \brief Do a synchronize copy from a continugous CPU memory region.
  *
  *  This function will call WaitToWrite before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copy from.
  * \param size the memory size we want to copy from.
  */
  void SyncCopyFromCPU(const void *data, size_t size = 0);
  /*!
  * \brief Do a synchronize copy from a continugous CPU memory region.
  *
  *  This function will call WaitToWrite before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copy from, int the form of T vector
  */
  template <typename T>
  void SyncCopyFromCPU(const std::vector<T> &data);
  /*!
  * \brief Do a synchronize copy to a continugous CPU memory region.
  *
  *  This function will call WaitToRead before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copyinto.
  * \param size the memory size we want to copy into. Defualt value is Size()
  */
  template <typename T>
  void SyncCopyToCPU(T *data, size_t size = 0);
  /*!
  * \brief Do a synchronize copy to a continugous CPU memory region.
  *
  *  This function will call WaitToRead before the copy is performed.
  *  This is useful to copy data from existing memory region that are
  *  not wrapped by NDArray(thus dependency not being tracked).
  *
  * \param data the data source to copyinto.
  * \param size the memory size we want to copy into. Defualt value is Size()
  */
  template <typename T>
  void SyncCopyToCPU(std::vector<T> &data);
  /*!
  * \brief Copy the content of current array to other.
  * \param other the new context of this NDArray
  * \return the new copy
  */
  NDArray CopyTo(NDArray * other) const;
  /*!
  * \brief return a new copy this NDArray
  * \param other the target NDArray
  * \return the copy target NDarray
  */
  NDArray Copy(const Context &) const;
  /*!
  * \brief return offset of the element at (h, w)
  * \param h height position
  * \param w width position
  * \return offset of two dimensions array
  */
  size_t Offset(size_t h = 0, size_t w = 0) const;
  /*!
   * \brief return offset of three dimensions array
   * \param c channel position
   * \param h height position
   * \param w width position
   * \return offset of three dimensions array
   */
  size_t Offset(size_t c, size_t h, size_t w) const;
  /*!
  * \brief return value of the element at (h, w)
  * \param h height position
  * \param w width position
  * \return value of two dimensions array
  */
  template <typename T>
  inline T At(size_t h, size_t w) const {
    return ((T*)GetData())[Offset(h, w)];
  }
  /*!
   * \brief return value of three dimensions array
   * \param c channel position
   * \param h height position
   * \param w width position
   * \return value of three dimensions array
   */
  template <typename T>
  T At(size_t c, size_t h, size_t w) const {
    return ((T*)GetData())[Offset(c, h, w)];
  }
  /*!
  * \brief Slice a NDArray
  * \param begin begin index in first dim
  * \param end end index in first dim
  * \return sliced NDArray
  */
  NDArray Slice(mx_uint begin, mx_uint end) const;
  /*!
  * \brief Return a reshaped NDArray that shares memory with current one
  * \param new_shape the new shape
  * \return reshaped NDarray
  */
  NDArray Reshape(const Shape &new_shape) const;
  /*!
  * \brief Block until all the pending write operations with respect
  *    to current NDArray are finished, and read can be performed.
  */
  void WaitToRead() const;
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and write can be performed.
  */
  void WaitToWrite();
  /*!
  * \brief Block until all the pending read/write operations with respect
  *    to current NDArray are finished, and read/write can be performed.
  */
  static void WaitAll();
  /*!
  * \brief Sample gaussian distribution for each elements of out.
  * \param mu mean of gaussian distribution.
  * \param sigma standard deviation of gaussian distribution.
  * \param out output NDArray.
  */
  static void SampleGaussian(mx_float mu, mx_float sigma, NDArray *out);
  /*!
  * \brief Sample uniform distribution for each elements of out.
  * \param begin lower bound of distribution.
  * \param end upper bound of distribution.
  * \param out output NDArray.
  */
  static void SampleUniform(mx_float begin, mx_float end, NDArray *out);
  /*!
  * \brief Load NDArrays from binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays returned, do not fill the list if
  * nullptr is given.
  * \param array_map a map from names to NDArrays returned, do not fill the map
  * if nullptr is given or no names is stored in binary file.
  */
  static void Load(const void* param_data, size_t param_size,
                   std::vector<NDArray> *array_list = nullptr,
                   std::map<std::string, NDArray> *array_map = nullptr);
  /*!
  * \brief Load map of NDArrays from binary file.
  * \param file_name name of the binary file.
  * \return a list of NDArrays.
  */
  static std::map<std::string, NDArray> LoadToMap(const void* param_data, size_t param_size);
  /*!
  * \brief Load list of NDArrays from binary file.
  * \param file_name name of the binary file.
  * \return a map from names to NDArrays.
  */
  static std::vector<NDArray> LoadToList(const void* param_data, size_t param_size);
  /*!
  * \brief save a map of string->NDArray to binary file.
  * \param file_name name of the binary file.
  * \param array_map a map from names to NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::map<std::string, NDArray> &array_map);
  /*!
  * \brief save a list of NDArrays to binary file.
  * \param file_name name of the binary file.
  * \param array_list a list of NDArrays.
  */
  static void Save(const std::string &file_name,
                   const std::vector<NDArray> &array_list);
  /*!
  * \return the size of current NDArray, a.k.a. the production of all shape dims
  */
  size_t Size() const;
  /*!
  * \return the shape of current NDArray, in the form of mx_uint vector
  */
  std::vector<mx_uint> GetShape() const;
  /*!
  * \return the data type of current NDArray
  */
  int GetDType() const;
  /*!
  * \return the data pointer to the current NDArray
  */
  const void *GetData() const;

  /*!
  * \return the context of NDArray
  */
  Context GetContext() const;

  /*!
  * \return the NDArrayHandle of the current NDArray
  */
  NDArrayHandle GetHandle() const { return blob_ptr_->handle_; }

 private:
  std::shared_ptr<NDBlob> blob_ptr_;
};
}  // namespace cpp
}  // namespace mxnet

#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_NDARRAY_H_
