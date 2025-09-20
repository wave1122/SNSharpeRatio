#ifndef ASSERT_H
#define ASSERT_H
#define DEBUG

#include <iostream>

#ifndef DEBUG
    #define ASSERT_(x)
#else
    #define ASSERT_(x) \
            if (! (x)) \
			{ \
                cout << "ERROR!!! Assert " << #x << " failed\n"; \
                cout << " on line " << __LINE__  << "\n";  \
                cout << " in file " << __FILE__ << "\n";  \
            }
#endif

#endif
