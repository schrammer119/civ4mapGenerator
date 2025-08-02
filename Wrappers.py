import time

def profile(func):
    """
    A simple decorator that profiles function calls with one-line output.

    Usage:
        @profile
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        # Get function info
        func_name = getattr(func, 'func_name', getattr(func, '__name__', 'unknown'))

        # Handle class methods
        if args and hasattr(args[0], func_name):
            class_name = args[0].__class__.__name__
            display_name = class_name + '.' + func_name
        else:
            display_name = func_name

        # Start timing
        start_time = time.time()

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Print one-line summary
            print ("%60s: %8.3f seconds" % (display_name, execution_time))

            return result

        except Exception:
            # Calculate execution time even on failure
            end_time = time.time()
            execution_time = end_time - start_time

            # Print one-line summary for failed calls
            print ("%60s: %8.3f seconds (FAILED)" % (display_name, execution_time))

            raise  # Re-raise the exception

    # Copy basic function attributes
    wrapper.__name__ = getattr(func, '__name__', 'wrapped_function')
    wrapper.__doc__ = getattr(func, '__doc__', None)

    return wrapper


# Example usage
if __name__ == "__main__":
    @profile
    def slow_function(n):
        """A function that takes some time to execute."""
        time.sleep(0.1)
        total = 0
        for i in range(n):
            total += i
        return total

    @profile
    def fast_function():
        """A fast function."""
        return 42

    class Calculator:
        @profile
        def add(self, a, b):
            return a + b

        @profile
        def slow_multiply(self, a, b):
            time.sleep(0.05)
            return a * b

    # Test the profiling
    print ("Testing function profiling:")
    result1 = slow_function(1000)
    result2 = fast_function()

    print ("\nTesting class method profiling:")
    calc = Calculator()
    sum_result = calc.add(5, 3)
    mult_result = calc.slow_multiply(4, 7)