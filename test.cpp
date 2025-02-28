#include <iostream>
#include <string>

// 基类
class Animal {
private:
    std::string name;
    int age;

public:
    // 基类构造函数
    Animal(const std::string& name, int age) : name(name), age(age) {
        std::cout << "Animal 构造函数执行: 创建 " << name << "，年龄 " << age << std::endl;
    }
    
    std::string getName() const { return name; }
    int getAge() const { return age; }
};

// 派生类
class Dog : public Animal {
private:
    std::string breed;

public:
    // 派生类构造函数，通过初始化列表调用基类构造函数
    Dog(const std::string& name, int age, const std::string& breed) 
        : Animal(name, age), // 调用基类构造函数
          breed(breed)       // 初始化派生类成员
    {
        // 派生类构造函数主体
        std::cout << "Dog 构造函数执行: " << getName() << " 是一只 " << breed << " 品种" << std::endl;
    }

    std::string getBreed() const { return breed; }
};

int main() {
    std::cout << "开始创建Dog对象..." << std::endl;
    
    // 创建Dog对象时，会先调用Animal构造函数，再执行Dog构造函数主体
    Dog myDog("旺财", 3, "金毛寻回犬");
    
    std::cout << "\n对象信息摘要:" << std::endl;
    std::cout << "名字: " << myDog.getName() << std::endl;
    std::cout << "年龄: " << myDog.getAge() << std::endl;
    std::cout << "品种: " << myDog.getBreed() << std::endl;
    
    return 0;
}