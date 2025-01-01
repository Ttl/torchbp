// -*- C++ -*-
//===--------------------------- complex ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_COMPLEX
#define _LIBCUDACXX_COMPLEX

/*
    complex synopsis

namespace std
{

template<class T>
class complex
{
public:
    typedef T value_type;

    complex(const T& re = T(), const T& im = T()); // constexpr in C++14
    complex(const complex&);  // constexpr in C++14
    template<class X> complex(const complex<X>&);  // constexpr in C++14

    T real() const; // constexpr in C++14
    T imag() const; // constexpr in C++14

    void real(T);
    void imag(T);

    complex<T>& operator= (const T&);
    complex<T>& operator+=(const T&);
    complex<T>& operator-=(const T&);
    complex<T>& operator*=(const T&);
    complex<T>& operator/=(const T&);

    complex& operator=(const complex&);
    template<class X> complex<T>& operator= (const complex<X>&);
    template<class X> complex<T>& operator+=(const complex<X>&);
    template<class X> complex<T>& operator-=(const complex<X>&);
    template<class X> complex<T>& operator*=(const complex<X>&);
    template<class X> complex<T>& operator/=(const complex<X>&);
};

template<>
class complex<float>
{
public:
    typedef float value_type;

    constexpr complex(float re = 0.0f, float im = 0.0f);
    explicit constexpr complex(const complex<double>&);
    explicit constexpr complex(const complex<long double>&);

    constexpr float real() const;
    void real(float);
    constexpr float imag() const;
    void imag(float);

    complex<float>& operator= (float);
    complex<float>& operator+=(float);
    complex<float>& operator-=(float);
    complex<float>& operator*=(float);
    complex<float>& operator/=(float);

    complex<float>& operator=(const complex<float>&);
    template<class X> complex<float>& operator= (const complex<X>&);
    template<class X> complex<float>& operator+=(const complex<X>&);
    template<class X> complex<float>& operator-=(const complex<X>&);
    template<class X> complex<float>& operator*=(const complex<X>&);
    template<class X> complex<float>& operator/=(const complex<X>&);
};

template<>
class complex<double>
{
public:
    typedef double value_type;

    constexpr complex(double re = 0.0, double im = 0.0);
    constexpr complex(const complex<float>&);
    explicit constexpr complex(const complex<long double>&);

    constexpr double real() const;
    void real(double);
    constexpr double imag() const;
    void imag(double);

    complex<double>& operator= (double);
    complex<double>& operator+=(double);
    complex<double>& operator-=(double);
    complex<double>& operator*=(double);
    complex<double>& operator/=(double);
    complex<double>& operator=(const complex<double>&);

    template<class X> complex<double>& operator= (const complex<X>&);
    template<class X> complex<double>& operator+=(const complex<X>&);
    template<class X> complex<double>& operator-=(const complex<X>&);
    template<class X> complex<double>& operator*=(const complex<X>&);
    template<class X> complex<double>& operator/=(const complex<X>&);
};

template<>
class complex<long double>
{
public:
    typedef long double value_type;

    constexpr complex(long double re = 0.0L, long double im = 0.0L);
    constexpr complex(const complex<float>&);
    constexpr complex(const complex<double>&);

    constexpr long double real() const;
    void real(long double);
    constexpr long double imag() const;
    void imag(long double);

    complex<long double>& operator=(const complex<long double>&);
    complex<long double>& operator= (long double);
    complex<long double>& operator+=(long double);
    complex<long double>& operator-=(long double);
    complex<long double>& operator*=(long double);
    complex<long double>& operator/=(long double);

    template<class X> complex<long double>& operator= (const complex<X>&);
    template<class X> complex<long double>& operator+=(const complex<X>&);
    template<class X> complex<long double>& operator-=(const complex<X>&);
    template<class X> complex<long double>& operator*=(const complex<X>&);
    template<class X> complex<long double>& operator/=(const complex<X>&);
};

// 26.3.6 operators:
template<class T> complex<T> operator+(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&, const T&);
template<class T> complex<T> operator+(const T&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const T&);
template<class T> complex<T> operator-(const T&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const T&);
template<class T> complex<T> operator*(const T&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const T&);
template<class T> complex<T> operator/(const T&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&);
template<class T> bool operator==(const complex<T>&, const complex<T>&); // constexpr in C++14
template<class T> bool operator==(const complex<T>&, const T&); // constexpr in C++14
template<class T> bool operator==(const T&, const complex<T>&); // constexpr in C++14
template<class T> bool operator!=(const complex<T>&, const complex<T>&); // constexpr in C++14
template<class T> bool operator!=(const complex<T>&, const T&); // constexpr in C++14
template<class T> bool operator!=(const T&, const complex<T>&); // constexpr in C++14

template<class T, class charT, class traits>
  basic_istream<charT, traits>&
  operator>>(basic_istream<charT, traits>&, complex<T>&);
template<class T, class charT, class traits>
  basic_ostream<charT, traits>&
  operator<<(basic_ostream<charT, traits>&, const complex<T>&);

// 26.3.7 values:

template<class T>              T real(const complex<T>&); // constexpr in C++14
                     long double real(long double);       // constexpr in C++14
                          double real(double);            // constexpr in C++14
template<Integral T>      double real(T);                 // constexpr in C++14
                          float  real(float);             // constexpr in C++14

template<class T>              T imag(const complex<T>&); // constexpr in C++14
                     long double imag(long double);       // constexpr in C++14
                          double imag(double);            // constexpr in C++14
template<Integral T>      double imag(T);                 // constexpr in C++14
                          float  imag(float);             // constexpr in C++14

template<class T> T abs(const complex<T>&);

template<class T>              T arg(const complex<T>&);
                     long double arg(long double);
                          double arg(double);
template<Integral T>      double arg(T);
                          float  arg(float);

template<class T>              T norm(const complex<T>&);
                     long double norm(long double);
                          double norm(double);
template<Integral T>      double norm(T);
                          float  norm(float);

template<class T>      complex<T>           conj(const complex<T>&);
                       complex<long double> conj(long double);
                       complex<double>      conj(double);
template<Integral T>   complex<double>      conj(T);
                       complex<float>       conj(float);

template<class T>    complex<T>           proj(const complex<T>&);
                     complex<long double> proj(long double);
                     complex<double>      proj(double);
template<Integral T> complex<double>      proj(T);
                     complex<float>       proj(float);

template<class T> complex<T> polar(const T&, const T& = T());

// 26.3.8 transcendentals:
template<class T> complex<T> acos(const complex<T>&);
template<class T> complex<T> asin(const complex<T>&);
template<class T> complex<T> atan(const complex<T>&);
template<class T> complex<T> acosh(const complex<T>&);
template<class T> complex<T> asinh(const complex<T>&);
template<class T> complex<T> atanh(const complex<T>&);
template<class T> complex<T> cos (const complex<T>&);
template<class T> complex<T> cosh (const complex<T>&);
template<class T> complex<T> exp (const complex<T>&);
template<class T> complex<T> log (const complex<T>&);
template<class T> complex<T> log10(const complex<T>&);

template<class T> complex<T> pow(const complex<T>&, const T&);
template<class T> complex<T> pow(const complex<T>&, const complex<T>&);
template<class T> complex<T> pow(const T&, const complex<T>&);

template<class T> complex<T> sin (const complex<T>&);
template<class T> complex<T> sinh (const complex<T>&);
template<class T> complex<T> sqrt (const complex<T>&);
template<class T> complex<T> tan (const complex<T>&);
template<class T> complex<T> tanh (const complex<T>&);

template<class T, class charT, class traits>
  basic_istream<charT, traits>&
  operator>>(basic_istream<charT, traits>& is, complex<T>& x);

template<class T, class charT, class traits>
  basic_ostream<charT, traits>&
  operator<<(basic_ostream<charT, traits>& o, const complex<T>& x);

}  // std

*/

#ifndef __cuda_std__
#include <__config>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <version>
#include <__pragma_push>
#endif //__cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

# if _LIBCUDACXX_CUDA_ABI_VERSION > 3
#  define _LIBCUDACXX_COMPLEX_ALIGNAS(V) _ALIGNAS(V)
# else
#  define _LIBCUDACXX_COMPLEX_ALIGNAS(V)
# endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template<class _Tp> class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(2*sizeof(_Tp)) complex;

template<class _Tp> _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);

template<class _Tp> _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template<class _Tp>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(2*sizeof(_Tp)) complex
{
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    complex(const value_type& __re = value_type(), const value_type& __im = value_type())
        : __re_(__re), __im_(__im) {}
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 value_type real() const {return __re_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 value_type imag() const {return __im_;}

    _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re) {__re_ = __re;}
    _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im) {__im_ = __im;}

    _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (const value_type& __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const value_type& __re) {__re_ += __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const value_type& __re) {__re_ -= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const value_type& __re) {__re_ *= __re; __im_ *= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const value_type& __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<> class complex<double>;
#ifdef _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
template<> class complex<long double>;
#endif // _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE

template<>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(2*sizeof(float)) complex<float>
{
    float __re_;
    float __im_;
public:
    typedef float value_type;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR complex(float __re = 0.0f, float __im = 0.0f)
        : __re_(__re), __im_(__im) {}
    _LIBCUDACXX_INLINE_VISIBILITY
    explicit _LIBCUDACXX_CONSTEXPR complex(const complex<double>& __c);
    _LIBCUDACXX_INLINE_VISIBILITY
    explicit _LIBCUDACXX_CONSTEXPR complex(const complex<long double>& __c);

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR float real() const {return __re_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR float imag() const {return __im_;}

    _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re) {__re_ = __re;}
    _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im) {__im_ = __im;}

    _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (float __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(float __re) {__re_ += __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(float __re) {__re_ -= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(float __re) {__re_ *= __re; __im_ *= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(float __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(2*sizeof(double)) complex<double>
{
    double __re_;
    double __im_;
public:
    typedef double value_type;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR complex(double __re = 0.0, double __im = 0.0)
        : __re_(__re), __im_(__im) {}
    _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR complex(const complex<float>& __c);
    _LIBCUDACXX_INLINE_VISIBILITY
    explicit _LIBCUDACXX_CONSTEXPR complex(const complex<long double>& __c);

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR double real() const {return __re_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR double imag() const {return __im_;}

    _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re) {__re_ = __re;}
    _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im) {__im_ = __im;}

    _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(double __re) {__re_ += __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(double __re) {__re_ -= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(double __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(2*sizeof(long double)) complex<long double>
{
#ifndef _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
public:
    template <typename _Dummy = void>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR complex(long double __re = 0.0, long double __im = 0.0)
        {static_assert(is_same<_Dummy, void>::value, "complex<long double> is not supported");}

    template <typename _Tp, typename _Dummy = void>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR complex(const complex<_Tp> &__c)
        {static_assert(is_same<_Dummy, void>::value, "complex<long double> is not supported");}

#else
    long double __re_;
    long double __im_;
public:
    typedef long double value_type;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR complex(long double __re = 0.0L, long double __im = 0.0L)
        : __re_(__re), __im_(__im) {}
    _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR complex(const complex<float>& __c);
    _LIBCUDACXX_INLINE_VISIBILITY
    _LIBCUDACXX_CONSTEXPR complex(const complex<double>& __c);

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR long double real() const {return __re_;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR long double imag() const {return __im_;}

    _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re) {__re_ = __re;}
    _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im) {__im_ = __im;}

    _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (long double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(long double __re) {__re_ += __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(long double __re) {__re_ -= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(long double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(long double __re) {__re_ /= __re; __im_ /= __re; return *this;}
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
#endif // _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
};

#if defined(_LIBCUDACXX_USE_PRAGMA_MSVC_WARNING)
  // MSVC complains about narrowing conversions on these copy constructors regardless if they are used
  #pragma warning(push)
  #pragma warning(disable : 4244)
#endif

inline
_LIBCUDACXX_CONSTEXPR
complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
_LIBCUDACXX_CONSTEXPR
complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

#ifdef _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
inline
_LIBCUDACXX_CONSTEXPR
complex<float>::complex(const complex<long double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
_LIBCUDACXX_CONSTEXPR
complex<double>::complex(const complex<long double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
_LIBCUDACXX_CONSTEXPR
complex<long double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
_LIBCUDACXX_CONSTEXPR
complex<long double>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}
#endif // _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE

#if defined(_LIBCUDACXX_USE_PRAGMA_MSVC_WARNING)
  #pragma warning(pop)
#endif

// 26.3.6 operators:

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t += __x;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(-__y);
    __t += __x;
    return __t;
}

template<class _Tp>
complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    /*
    if (__libcpp_isnan_or_builtin(__x) && __libcpp_isnan_or_builtin(__y))
    {
        bool __recalc = false;
        if (__libcpp_isinf_or_builtin(__a) || __libcpp_isinf_or_builtin(__b))
        {
            __a = copysign(__libcpp_isinf_or_builtin(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(__libcpp_isinf_or_builtin(__b) ? _Tp(1) : _Tp(0), __b);
            if (__libcpp_isnan_or_builtin(__c))
                __c = copysign(_Tp(0), __c);
            if (__libcpp_isnan_or_builtin(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__libcpp_isinf_or_builtin(__c) || __libcpp_isinf_or_builtin(__d))
        {
            __c = copysign(__libcpp_isinf_or_builtin(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(__libcpp_isinf_or_builtin(__d) ? _Tp(1) : _Tp(0), __d);
            if (__libcpp_isnan_or_builtin(__a))
                __a = copysign(_Tp(0), __a);
            if (__libcpp_isnan_or_builtin(__b))
                __b = copysign(_Tp(0), __b);
            __recalc = true;
        }
        if (!__recalc && (__libcpp_isinf_or_builtin(__ac) || __libcpp_isinf_or_builtin(__bd) ||
                          __libcpp_isinf_or_builtin(__ad) || __libcpp_isinf_or_builtin(__bc)))
        {
            if (__libcpp_isnan_or_builtin(__a))
                __a = copysign(_Tp(0), __a);
            if (__libcpp_isnan_or_builtin(__b))
                __b = copysign(_Tp(0), __b);
            if (__libcpp_isnan_or_builtin(__c))
                __c = copysign(_Tp(0), __c);
            if (__libcpp_isnan_or_builtin(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__recalc)
        {
            __x = _Tp(INFINITY) * (__a * __c - __b * __d);
            __y = _Tp(INFINITY) * (__a * __d + __b * __c);
        }
    }
    */
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator*(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t *= __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator*(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t *= __x;
    return __t;
}

namespace detail {
    template <class _Tp>
    inline _LIBCUDACXX_INLINE_VISIBILITY
    _Tp __scalbn(_Tp __x, int __i) {
        return static_cast<_Tp>(scalbn(static_cast<double>(__x), __i));
    }

    template <>
    inline _LIBCUDACXX_INLINE_VISIBILITY
    float __scalbn<float>(float __x, int __i) {
        return scalbnf(__x, __i);
    }

    template <>
    inline _LIBCUDACXX_INLINE_VISIBILITY
    double __scalbn<double>(double __x, int __i) {
        return scalbn(__x, __i);
    }

#ifndef _LIBCUDACXX_COMPILER_NVRTC
    template <>
    inline _LIBCUDACXX_INLINE_VISIBILITY
    long double __scalbn<long double>(long double __x, int __i) {
        return scalbnl(__x, __i);
    }
#endif
}

template<class _Tp>
complex<_Tp>
operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    int __ilogbw = 0;
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
    if (__libcpp_isfinite_or_builtin(__logbw))
    {
        __ilogbw = static_cast<int>(__logbw);
        __c = detail::__scalbn(__c, -__ilogbw);
        __d = detail::__scalbn(__d, -__ilogbw);
    }
    _Tp __denom = __c * __c + __d * __d;
    _Tp __x = detail::__scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    _Tp __y = detail::__scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    /*
    if (__libcpp_isnan_or_builtin(__x) && __libcpp_isnan_or_builtin(__y))
    {
        if ((__denom == _Tp(0)) && (!__libcpp_isnan_or_builtin(__a) || !__libcpp_isnan_or_builtin(__b)))
        {
            __x = copysign(_Tp(INFINITY), __c) * __a;
            __y = copysign(_Tp(INFINITY), __c) * __b;
        }
        else if ((__libcpp_isinf_or_builtin(__a) || __libcpp_isinf_or_builtin(__b)) && __libcpp_isfinite_or_builtin(__c) && __libcpp_isfinite_or_builtin(__d))
        {
            __a = copysign(__libcpp_isinf_or_builtin(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(__libcpp_isinf_or_builtin(__b) ? _Tp(1) : _Tp(0), __b);
            __x = _Tp(INFINITY) * (__a * __c + __b * __d);
            __y = _Tp(INFINITY) * (__b * __c - __a * __d);
        }
        else if (__libcpp_isinf_or_builtin(__logbw) && __logbw > _Tp(0) && __libcpp_isfinite_or_builtin(__a) && __libcpp_isfinite_or_builtin(__b))
        {
            __c = copysign(__libcpp_isinf_or_builtin(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(__libcpp_isinf_or_builtin(__d) ? _Tp(1) : _Tp(0), __d);
            __x = _Tp(0) * (__a * __c + __b * __d);
            __y = _Tp(0) * (__b * __c - __a * __d);
        }
    }
    */
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator/(const complex<_Tp>& __x, const _Tp& __y)
{
    return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator/(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t /= __y;
    return __t;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator+(const complex<_Tp>& __x)
{
    return __x;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
operator-(const complex<_Tp>& __x)
{
    return complex<_Tp>(-__x.real(), -__x.imag());
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    return __x.real() == __y && __x.imag() == 0;
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    return __x == __y.real() && 0 == __y.imag();
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
                     bool = is_floating_point<_Tp>::value
                     >
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, true, false>
{
    typedef double _ValueType;
    typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, false, true>
{
    typedef _Tp _ValueType;
    typedef complex<_Tp> _ComplexType;
};

// real

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
_Tp
real(const complex<_Tp>& __c)
{
    return __c.real();
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
real(_Tp __re)
{
    return __re;
}

// imag

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
_Tp
imag(const complex<_Tp>& __c)
{
    return __c.imag();
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
imag(_Tp)
{
    return 0;
}

// abs

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_Tp
abs(const complex<_Tp>& __c)
{
    return hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_Tp
arg(const complex<_Tp>& __c)
{
    return atan2(__c.imag(), __c.real());
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if<
    is_same<_Tp, long double>::value,
    long double
>::type
arg(_Tp __re)
{
    return atan2l(0.L, __re);
}

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if
<
    is_integral<_Tp>::value || is_same<_Tp, double>::value,
    double
>::type
arg(_Tp __re)
{
    // integrals need to be promoted to double
    return atan2(0., static_cast<double>(__re));
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if<
    is_same<_Tp, float>::value,
    float
>::type
arg(_Tp __re)
{
    return atan2f(0.F, __re);
}

// norm

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_Tp
norm(const complex<_Tp>& __c)
{
    if (__libcpp_isinf_or_builtin(__c.real()))
        return abs(__c.real());
    if (__libcpp_isinf_or_builtin(__c.imag()))
        return abs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
norm(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
    return static_cast<_ValueType>(__re) * __re;
}

// conj

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
conj(const complex<_Tp>& __c)
{
    return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
conj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}



// proj

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
proj(const complex<_Tp>& __c)
{
    std::complex<_Tp> __r = __c;
    if (__libcpp_isinf_or_builtin(__c.real()) || __libcpp_isinf_or_builtin(__c.imag()))
        __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
    return __r;
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if
<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    if (__libcpp_isinf_or_builtin(__re))
        __re = abs(__re);
    return complex<_Tp>(__re);
}

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if
<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}

// polar

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
    if (__libcpp_isnan_or_builtin(__rho) || signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (__libcpp_isnan_or_builtin(__theta))
    {
        if (__libcpp_isinf_or_builtin(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (__libcpp_isinf_or_builtin(__theta))
    {
        if (__libcpp_isinf_or_builtin(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * cos(__theta);
    if (__libcpp_isnan_or_builtin(__x))
        __x = 0;
    _Tp __y = __rho * sin(__theta);
    if (__libcpp_isnan_or_builtin(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
log(const complex<_Tp>& __x)
{
    return complex<_Tp>(log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
log10(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(10));
}

// sqrt

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
sqrt(const complex<_Tp>& __x)
{
    if (__libcpp_isinf_or_builtin(__x.imag()))
        return complex<_Tp>(_Tp(INFINITY), __x.imag());
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (__x.real() > _Tp(0))
            return complex<_Tp>(__x.real(), __libcpp_isnan_or_builtin(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
        return complex<_Tp>(__libcpp_isnan_or_builtin(__x.imag()) ? __x.imag() : _Tp(0), copysign(__x.real(), __x.imag()));
    }
    return polar(sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
exp(const complex<_Tp>& __x)
{
    _Tp __i = __x.imag();
    if (__i == 0) {
        return complex<_Tp>(exp(__x.real()), copysign(_Tp(0), __x.imag()));
    }
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (__x.real() < _Tp(0))
        {
            if (!__libcpp_isfinite_or_builtin(__i))
                __i = _Tp(1);
        }
        else if (__i == 0 || !__libcpp_isfinite_or_builtin(__i))
        {
            if (__libcpp_isinf_or_builtin(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    _Tp __e = exp(__x.real());
    return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return exp(__y * log(__x));
}

template<class _Tp, class _Up>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<typename __promote<_Tp, _Up>::type>
pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return _CUDA_VSTD::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if
<
    is_arithmetic<_Up>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const complex<_Tp>& __x, const _Up& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return _CUDA_VSTD::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename enable_if
<
    is_arithmetic<_Tp>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const _Tp& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return _CUDA_VSTD::pow(result_type(__x), result_type(__y));
}

// __sqr, computes pow(x, 2)

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
__sqr(const complex<_Tp>& __x)
{
    return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                        _Tp(2) * __x.real() * __x.imag());
}

// asinh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
asinh(const complex<_Tp>& __x)
{
    const _Tp __pi(static_cast<_Tp>(atan2(+0., -0.)));
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (__libcpp_isnan_or_builtin(__x.imag()))
            return __x;
        if (__libcpp_isinf_or_builtin(__x.imag()))
            return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (__libcpp_isnan_or_builtin(__x.real()))
    {
        if (__libcpp_isinf_or_builtin(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__x.imag() == 0)
            return __x;
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (__libcpp_isinf_or_builtin(__x.imag()))
        return complex<_Tp>(copysign(__x.imag(), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
acosh(const complex<_Tp>& __x)
{
    const _Tp __pi(static_cast<_Tp>(atan2(+0., -0.)));
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (__libcpp_isnan_or_builtin(__x.imag()))
            return complex<_Tp>(abs(__x.real()), __x.imag());
        if (__libcpp_isinf_or_builtin(__x.imag()))
        {
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
            else
                return complex<_Tp>(-__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
        }
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (__libcpp_isnan_or_builtin(__x.real()))
    {
        if (__libcpp_isinf_or_builtin(__x.imag()))
            return complex<_Tp>(abs(__x.imag()), __x.real());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (__libcpp_isinf_or_builtin(__x.imag()))
        return complex<_Tp>(abs(__x.imag()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
}

// atanh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
atanh(const complex<_Tp>& __x)
{
    const _Tp __pi(static_cast<_Tp>(atan2(+0., -0.)));
    if (__libcpp_isinf_or_builtin(__x.imag()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (__libcpp_isnan_or_builtin(__x.imag()))
    {
        if (__libcpp_isinf_or_builtin(__x.real()) || __x.real() == 0)
            return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }
    if (__libcpp_isnan_or_builtin(__x.real()))
    {
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
    {
        return complex<_Tp>(copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
    }
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
sinh(const complex<_Tp>& __x)
{
    if (__libcpp_isinf_or_builtin(__x.real()) && !__libcpp_isfinite_or_builtin(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.real() == 0 && !__libcpp_isfinite_or_builtin(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.imag() == 0 && !__libcpp_isfinite_or_builtin(__x.real()))
        return __x;
    return complex<_Tp>(sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
cosh(const complex<_Tp>& __x)
{
    if (__libcpp_isinf_or_builtin(__x.real()) && !__libcpp_isfinite_or_builtin(__x.imag()))
        return complex<_Tp>(abs(__x.real()), _Tp(NAN));
    if (__x.real() == 0 && !__libcpp_isfinite_or_builtin(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());
    if (__x.imag() == 0 && !__libcpp_isfinite_or_builtin(__x.real()))
        return complex<_Tp>(abs(__x.real()), __x.imag());
    return complex<_Tp>(cosh(__x.real()) * cos(__x.imag()), sinh(__x.real()) * sin(__x.imag()));
}

// tanh

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
tanh(const complex<_Tp>& __x)
{
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (!__libcpp_isfinite_or_builtin(__x.imag()))
            return complex<_Tp>(copysign(_Tp(1), __x.real()), _Tp(0));
        return complex<_Tp>(copysign(_Tp(1), __x.real()), copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
    }
    if (__libcpp_isnan_or_builtin(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __2r(_Tp(2) * __x.real());
    _Tp __2i(_Tp(2) * __x.imag());
    _Tp __d(cosh(__2r) + cos(__2i));
    _Tp __2rsh(sinh(__2r));
    if (__libcpp_isinf_or_builtin(__2rsh) && __libcpp_isinf_or_builtin(__d))
        return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                            __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
    return  complex<_Tp>(__2rsh/__d, sin(__2i)/__d);
}

// asin

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
asin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
acos(const complex<_Tp>& __x)
{
    const _Tp __pi(static_cast<_Tp>(atan2(+0., -0.)));
    if (__libcpp_isinf_or_builtin(__x.real()))
    {
        if (__libcpp_isnan_or_builtin(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__libcpp_isinf_or_builtin(__x.imag()))
        {
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
        return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
    }
    if (__libcpp_isnan_or_builtin(__x.real()))
    {
        if (__libcpp_isinf_or_builtin(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (__libcpp_isinf_or_builtin(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    if (signbit(__x.imag()))
        return complex<_Tp>(abs(__z.imag()), abs(__z.real()));
    return complex<_Tp>(abs(__z.imag()), -abs(__z.real()));
}

// atan

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
atan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
sin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template<class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
complex<_Tp>
cos(const complex<_Tp>& __x)
{
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY complex<_Tp>
tan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

#ifndef __cuda_std__

template<class _Tp, class _CharT, class _Traits>
basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, complex<_Tp>& __x)
{
    if (__is.good())
    {
        ws(__is);
        if (__is.peek() == _CharT('('))
        {
            __is.get();
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
            {
                ws(__is);
                _CharT __c = __is.peek();
                if (__c == _CharT(','))
                {
                    __is.get();
                    _Tp __i;
                    __is >> __i;
                    if (!__is.fail())
                    {
                        ws(__is);
                        __c = __is.peek();
                        if (__c == _CharT(')'))
                        {
                            __is.get();
                            __x = complex<_Tp>(__r, __i);
                        }
                        else
                            __is.setstate(ios_base::failbit);
                    }
                    else
                        __is.setstate(ios_base::failbit);
                }
                else if (__c == _CharT(')'))
                {
                    __is.get();
                    __x = complex<_Tp>(__r, _Tp(0));
                }
                else
                    __is.setstate(ios_base::failbit);
            }
            else
                __is.setstate(ios_base::failbit);
        }
        else
        {
            _Tp __r;
            __is >> __r;
            if (!__is.fail())
                __x = complex<_Tp>(__r, _Tp(0));
            else
                __is.setstate(ios_base::failbit);
        }
    }
    else
        __is.setstate(ios_base::failbit);
    return __is;
}

template<class _Tp, class _CharT, class _Traits>
basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
{
    basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.real() << ',' << __x.imag() << ')';
    return __os << __s.str();
}

#endif // __cuda_std__

#if _LIBCUDACXX_STD_VER > 11 && defined(_LIBCUDACXX_HAS_STL_LITERALS)
// Literal suffix for complex number literals [complex.literals]
inline namespace literals
{
  inline namespace complex_literals
  {
#ifdef _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<long double> operator""il(long double __im)
    {
        return { 0.0l, __im };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<long double> operator""il(unsigned long long __im)
    {
        return { 0.0l, static_cast<long double>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<double> operator""i(long double __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<double> operator""i(unsigned long long __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<float> operator""if(long double __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<float> operator""if(unsigned long long __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }
#else
    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<double> operator""i(double __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<double> operator""i(unsigned long long __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<float> operator""if(double __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }

    _LIBCUDACXX_INLINE_VISIBILITY constexpr complex<float> operator""if(unsigned long long __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }
#endif
  }
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#ifndef __cuda_std__
#include <__pragma_pop>
#endif //__cuda_std__

#endif  // _LIBCUDACXX_COMPLEX
