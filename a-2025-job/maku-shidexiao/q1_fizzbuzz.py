class FizzBuzz(object):
    def __init__(self, start=1, end=100):
        self.start = start
        self.end = end

    def check_fizzfuzz_stage1(self, num):
        if num % 3 == 0 and num % 5 == 0:
            return "FizzBuzz"
        elif num % 3 == 0:
            return "Fizz"
        elif num % 5 == 0:
            return "Buzz"
        else:
            return str(num)

    def check_fizzfuzz_stage2(self, num):
        fizz_c = num % 3 == 0 or '3' in str(num)
        buzz_c = num % 5 == 0 or '5' in str(num)

        if fizz_c and buzz_c:
            return "FizzBuzz"
        elif fizz_c:
            return "Fizz"
        elif buzz_c:
            return "Buzz"
        else:
            return str(num)

    def play_stage1(self):
        for num in range(self.start, self.end+1):
            print(self.check_fizzfuzz_stage1(num))

    def play_stage2(self):
        for num in range(self.start, self.end+1):
            print(self.check_fizzfuzz_stage2(num))


if __name__ == "__main__":
    game = FizzBuzz()
    game.play_stage1()
    print('===')
    game.play_stage2()
