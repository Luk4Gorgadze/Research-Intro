def maximumCostSubstring(s, chars, vals):
        """
        :type s: str
        :type chars: str
        :type vals: List[int]
        :rtype: int
        """
        map = {}
        for i,c in enumerate(chars):
            map[c] = vals[i]
        
        R = 0
        rs = 0
        mx = 0
        while R < len(s):
            if s[R] not in map:
                rs += ord(s[R]) - ord("a") + 1
                mx = max(mx,rs)
            else:
                comp = rs + map[s[R]]
                if comp > 0:
                    rs = comp
                elif comp < 0:
                    rs = 0
                mx = max(mx,rs)
            R += 1
                    
                    
        
        return mx

print(maximumCostSubstring("hwwqwwqqqh",
"wihq",
[-2,-5,-4,4]))