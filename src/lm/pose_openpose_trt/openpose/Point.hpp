/*
 * @Author: zerollzeng
 * @Date: 2019-10-10 18:07:54
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-10 18:07:54
 */
#ifndef POINT_HPP
#define POINT_HPP


#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType  className<char>; \
    template classType  className<signed char>; \
    template classType  className<short>; \
    template classType  className<int>; \
    template classType  className<long>; \
    template classType  className<long long>; \
    template classType  className<unsigned char>; \
    template classType  className<unsigned short>; \
    template classType  className<unsigned int>; \
    template classType  className<unsigned long>; \
    template classType  className<unsigned long long>; \
    template classType  className<float>; \
    template classType  className<double>; \
    template classType  className<long double>

#define OVERLOAD_C_OUT(className) \
    template<typename T> std::ostream &operator<<(std::ostream& ostream, const op::className<T>& obj) \
    { \
        ostream << obj.toString(); \
        return ostream; \
    }

#include <string>

namespace op
{
    template<typename T>
    struct Point
    {
        T x;
        T y;

        Point(const T x = 0, const T y = 0);

        /**
         * Copy constructor.
         * It performs `fast copy`: For performance purpose, copying a Point<T> or Point<T> or cv::Mat just copies the
         * reference, it still shares the same internal data.
         * Modifying the copied element will modify the original one.
         * Use clone() for a slower but real copy, similarly to cv::Mat and Point<T>.
         * @param point Point to be copied.
         */
        Point<T>(const Point<T>& point);

        /**
         * Copy assignment.
         * Similar to Point<T>(const Point<T>& point).
         * @param point Point to be copied.
         * @return The resulting Point.
         */
        Point<T>& operator=(const Point<T>& point);

        /**
         * Move constructor.
         * It destroys the original Point to be moved.
         * @param point Point to be moved.
         */
        Point<T>(Point<T>&& point);

        /**
         * Move assignment.
         * Similar to Point<T>(Point<T>&& point).
         * @param point Point to be moved.
         * @return The resulting Point.
         */
        Point<T>& operator=(Point<T>&& point);

        inline T area() const
        {
            return x * y;
        }

        /**
         * It returns a string with the whole Point<T> data. Useful for debugging.
         * The format is: `[x, y]`
         * @return A string with the Point<T> values in the above format.
         */
        std::string toString() const;





        // ------------------------------ Comparison operators ------------------------------ //
        /**
         * Less comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator<(const Point<T>& point) const
        {
            return area() < point.area();
        }

        /**
         * Greater comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator>(const Point<T>& point) const
        {
            return area() > point.area();
        }

        /**
         * Less or equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator<=(const Point<T>& point) const
        {
            return area() <= point.area();
        }

        /**
         * Greater or equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator>=(const Point<T>& point) const
        {
            return area() >= point.area();
        }

        /**
         * Equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator==(const Point<T>& point) const
        {
            return area() == point.area();
        }

        /**
         * Not equal comparison operator.
         * @param point Point<T> to be compared.
         * @result Whether the instance satisfies the condition with respect to point.
         */
        inline bool operator!=(const Point<T>& point) const
        {
            return area() != point.area();
        }





        // ------------------------------ Basic Operators ------------------------------ //
        Point<T>& operator+=(const Point<T>& point);

        Point<T> operator+(const Point<T>& point) const;

        Point<T>& operator+=(const T value);

        Point<T> operator+(const T value) const;

        Point<T>& operator-=(const Point<T>& point);

        Point<T> operator-(const Point<T>& point) const;

        Point<T>& operator-=(const T value);

        Point<T> operator-(const T value) const;

        Point<T>& operator*=(const T value);

        Point<T> operator*(const T value) const;

        Point<T>& operator/=(const T value);

        Point<T> operator/(const T value) const;
    };

    // Static methods
    OVERLOAD_C_OUT(Point)
}

#endif // OPENPOSE_CORE_POINT_HPP
