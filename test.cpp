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
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        int count=0;
        int j=0;
        if(nums.size()==0||nums.size()==1)
        {}
        else
        {
        for(int i=nums.size()-2;i>=0;i--)
        {
        for(j=i+1;j<nums.size()&&nums[j]>=nums[i];j++);
        nums.insert(nums.cbegin()+j,nums[i]);
        nums.erase(nums.cbegin()+i);
        count+=(nums.size()-j);
        }
        }
        return count;
    }
};
