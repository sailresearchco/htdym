// distinct orderings: duplicate values collapse, so a multiset of primes
// permutes without repeats while distinct objects permute fully
export function permutations<T>(xs: T[]): T[][] {
  if (xs.length === 0) return [[]];
  return [...new Set(xs)].flatMap((first) => {
    const rest = [...xs];
    rest.splice(rest.indexOf(first), 1);
    return permutations(rest).map((tail) => [first, ...tail]);
  });
}

// every subset, the empty one included
export function subsets<T>(xs: T[]): T[][] {
  return xs.reduce<T[][]>((acc, x) => [...acc, ...acc.map((s) => [...s, x])], [[]]);
}

export function cross<T>(options: T[][]): T[][] {
  return options.reduce<T[][]>((acc, xs) => acc.flatMap((a) => xs.map((x) => [...a, x])), [[]]);
}

export function primes(n: number): number[] {
  const out: number[] = [];
  for (let p = 2; n > 1; p++)
    while (n % p === 0) {
      out.push(p);
      n /= p;
    }
  return out;
}

export function divisors(n: number): number[] {
  const out: number[] = [];
  for (let d = 1; d <= n; d++) if (n % d === 0) out.push(d);
  return out;
}
