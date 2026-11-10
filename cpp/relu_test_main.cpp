#include <gtest/gtest.h>

int ArgC = 0;
char** ArgV = nullptr;

int main(int argc, char** argv)
{
  ArgC = argc;
  ArgV = argv;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
