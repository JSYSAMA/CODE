//just a git test
#include<iostream>
#include<vector>
#include<algorithm>
#include<functional>

using namespace std;

int main()
{
    int ia[6]={27,210,12,47,109,83};
    vector<int,allocator<int>> vi(ia,ia+6);
    cout<<count_if(vi.begin(),vi.end(),not1(bind(less<int>(),placeholders::_1,40)));
    return 0;
}
