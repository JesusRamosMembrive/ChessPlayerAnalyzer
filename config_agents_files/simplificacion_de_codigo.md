
# Code Simplification Prompt

## Main Prompt

```
Please review all the code we just developed and simplify it following these criteria:

**MAIN OBJECTIVES:**
- Maintain exactly the same functionality
- Eliminate over-engineering and unnecessary abstractions
- Maximize readability and maintainability
- Reduce complexity without sacrificing robustness

**SIMPLIFICATION CRITERIA:**

1. **Remove premature abstractions:**
   - Remove interfaces, abstract classes, or patterns that don't add real value
   - Convert to simple functions what doesn't need to be a class
   - Eliminate unnecessary layers of indirection

2. **Reduce dependencies:**
   - Remove libraries or dependencies that are minimally used
   - Replace heavy dependencies with native solutions when simple
   - Consolidate similar functionalities

3. **Simplify structure:**
   - Reduce the number of files if possible without affecting organization
   - Eliminate excessive configurations
   - Simplify complex data structures

4. **Improve readability:**
   - Use clearer and more direct names
   - Remove obvious comments, keep only necessary ones
   - Reduce excessive nesting
   - Simplify complex conditionals

5. **Apply YAGNI (You Aren't Gonna Need It):**
   - Remove code for "future use cases" that aren't immediate
   - Remove excessive configurability
   - Remove unnecessary optional parameters

**WHAT YOU SHOULD NOT DO:**
- Don't remove important validations
- Don't sacrifice clarity for extreme brevity
- Don't remove necessary error handling
- Don't make the code fragile or hard to debug

**RESPONSE FORMAT:**
1. List the main changes you propose
2. Briefly explain why each change simplifies the code
3. Confirm that functionality remains intact
4. Provide the simplified code

Can you review the code with these criteria and propose a simplified version?
```

## Specific Variants

### For cases with heavy configuration:
```
Additional: Pay special attention to simplifying excessive configurations. 
If something can be safely hardcoded without losing real flexibility, do it.
```

### For cases with many abstractions:
```
Additional: Focus especially on eliminating abstractions that don't add value. 
If there's only one implementation of an interface, consider removing the interface.
```

### For cases with many dependencies:
```
Additional: Review if we can reduce external dependencies. 
Evaluate if any functionality can be implemented natively and simply.
```

## Usage Tips

1. **Timing:** Use it after completing a feature, not during development
2. **Iterative:** You can apply it multiple times if the first result still looks complex
3. **Context:** Mention if there are specific project constraints
4. **Review:** Always check that tests still pass after simplification