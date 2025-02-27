---
layout: ../layouts/Layout.astro
---

# Hello!

This is a test page with some Racket code:

```racket
#lang racket

(provide time-it)

(require (for-syntax syntax/parse))

(define-syntax (time-it stx)
  (syntax-parse stx
    [(_ task)
     #'(thunk-time-it (λ () task))]))

(define (thunk-time-it task)
  (define before (cim))
  (define answer (task))
  (define delta  (- (cim) before))
  (printf "time: ~a ms\n" delta)
  answer)

(define cim current-inexact-milliseconds)
```

Or maybe you'd like to see my Haskell λ-calculus parser?

```haskell
-- SPDX-FileCopyrightText: 2022 Severen Redwood <sev@severen.dev>
-- SPDX-License-Identifier: GPL-3.0-or-later

module Sly.Parser (parse) where

import Control.Monad.Combinators.Expr (Operator (..), makeExprParser)
import Data.Foldable (foldr')
import Data.Functor (void, ($>))
import Data.Text (Text)
import Data.Text qualified as T
import Data.Void (Void)
import Sly.Syntax
import Text.Megaparsec hiding (Token, parse)
import Text.Megaparsec.Char (newline)
import Text.Megaparsec.Char.Lexer qualified as L
import Unicode.Char.Identifiers (isPatternWhitespace, isXIDContinue, isXIDStart)

{- Grammar Notes
  * Application is parsed with the highest precedence and associativity to the
    left, i.e. f x y = ((f x) y).
  * Abstractions are parsed with bodies extending as far to the right as
    possible, i.e. λx -> λy -> x y x = (λx -> (λy -> ((x y) x))).
  * Bracketing of terms can be used to override precedence and associativity
    when required.
  * Statements are either an assignment (let X := Y) or a λ-calculus term and
    end with a full stop '.'.

  For anyone editing this file: One good test to flush out any bugs in the
  parser is to check whether the following equality is reflected by the parse
  trees of the LHS and RHS, respectively:
  λf -> (λx -> f (x x)) (λx -> f (x x)) = (λf -> ((λx -> (f (x x))) (λx -> (f (x x))))).
-}

-- | The parser monad.
--
-- This is a Parsec type synonym to both help type inference and the compiler's
-- optimiser.
type Parser = Parsec Void Text

-- | Words that are reserved as keywords and thus disallowed as names.
keywords :: [Text]
keywords = ["let", "in"]

-- | Parse and discard one or more whitespace characters.
space1 :: Parser ()
space1 = void $ some (satisfy isPatternWhitespace)

-- | Parse and discard zero or more whitespace characters and comments.
spaceConsumer :: Parser ()
spaceConsumer =
  L.space
    space1
    (L.skipLineComment "--")
    (L.skipBlockCommentNested "/-" "-/")

lexeme :: Parser a -> Parser a
lexeme = L.lexeme spaceConsumer

symbol :: Text -> Parser Text
symbol = L.symbol spaceConsumer

-- | Create a parser that will parse and discard the given string.
punc :: Text -> Parser ()
punc = void . symbol

-- | Create a parser that applies the given parser to an expression between a
-- pair of round brackets.
brackets :: Parser a -> Parser a
brackets = between (punc "(") (punc ")")

-- | Parse a 'start' character in a name.
nameStart :: Parser Char
nameStart = satisfy \c -> isXIDStart c && c /= 'λ'

-- | Parse a sequence of 'continue' characters in a name.
nameContinue :: Parser String
nameContinue = many $ satisfy \c -> isXIDContinue c || c == '\''

-- | Parse a name according to the Unicode Standard Annex #31.
name :: Parser Name
name = Name <$> (lexeme word >>= check) <?> "name"
  where
    word = T.pack <$> ((:) <$> nameStart <*> nameContinue)
    check w
      | w `notElem` keywords = return w
      -- TODO: See if the positioning of this error message (when output) can be
      -- improved.
      | otherwise = fail $ "keyword " <> T.unpack w <> " cannot be a name"

-- | Parse a variable term.
variable :: Parser Term
variable = Var <$> name

-- | Parse a natural number literal.
natural :: Parser Term
natural = toChurchNat <$> (lexeme L.decimal >>= check)
  where
    check n
      | n <= toInteger maxInt = return (fromInteger n)
      | otherwise = fail $ "naturals larger than " <> show maxInt <> " are disallowed"
    maxInt = maxBound @Int

-- | Parse a Boolean literal.
boolean :: Parser Term
boolean = toChurchBool <$> (single '#' *> (try true <|> false))
  where
    true = (symbol "true" <|> symbol "t") $> True
    false = (symbol "false" <|> symbol "f") $> False

-- | Parse a λ-abstraction.
abstraction :: Parser Term
abstraction = do
  punc "\\" <|> punc "λ" <?> "\\"
  binders <- name `sepBy1` spaceConsumer
  punc "->" <|> punc "↦" <?> "->"
  abstract binders <$> term
  where
    -- Expand an abstraction with multiple variables into its internal representation of
    -- nested single-variable abstractions.
    abstract = flip (foldr' Abs)

-- | Parse an application term.
application :: Parser (Term -> Term -> Term)
application = return App

-- | Parse the initial common fragment of a let term or a let statement.
letStart :: Parser (Name, Term)
letStart = do
  punc "let"
  name' <- name <?> "name"
  punc ":="
  term' <- term
  return (name', term')

-- | Parse a let term.
let_ :: Parser Term
let_ = do
  (x, t) <- letStart
  punc "in"
  body <- term
  -- NOTE: let x := t in body is syntactic sugar for (λx -> body) t.
  return $ App (Abs x body) t

-- | Parse a λ-term.
term :: Parser Term
term = makeExprParser (choice indivisibles) operatorTable
  where
    operatorTable = [[InfixL application]]
    indivisibles =
      [try variable, natural, boolean, abstraction, let_, brackets term]

-- | Parse an assignment statement.
assignment :: Parser Statement
assignment = do
  (x, t) <- letStart
  notFollowedBy "in"
  return (Ass x t)

-- | Parse a term statement.
termS :: Parser Statement
termS = Term <$> term

-- | Parse a program statement.
statement :: Parser Statement
statement = try assignment <|> termS

newlines :: Parser ()
newlines = void $ some newline

-- | Parse a complete program file.
--
--  We define 'program' in this context to be a sequence of bindings and/or terms.
program :: Parser [Statement]
program = spaceConsumer *> statements <* eof
  where
    statements = statement `sepEndBy` (spaceConsumer *> (newlines <|> punc ".")) <* spaceConsumer

-- | Parse a sly program given a filename and a program string.
parse :: FilePath -> Text -> Either (ParseErrorBundle Text Void) [Statement]
parse = runParser program
```

Here's some Lean code:

```lean
structure Point (α : Type u) where
  mk :: (x : α) (y : α)
  deriving Repr

#check Point       -- a Type
#check @Point.rec  -- the eliminator
#check @Point.mk   -- the constructor
#check @Point.x    -- a projection
#check @Point.y    -- a projection

#eval Point.x (Point.mk 10 20)
#eval Point.y (Point.mk 10 20)

open Point

example (a b : α) : x (mk a b) = a :=
  rfl

example (a b : α) : y (mk a b) = b :=
  rfl

def p := Point.mk 10 20

def Point.smul (n : Nat) (p : Point Nat) :=
  Point.mk (n * p.x) (n * p.y)

def xs : List Nat := [1, 2, 3]
def f : Nat → Nat := fun x => x * x

#eval xs.map f  -- [1, 4, 9]


structure MyStruct where
    {α : Type u}
    {β : Type v}
    a : α
    b : β

#check { a := 10, b := true : MyStruct }


structure Point (α : Type u) where
  x : α
  y : α
  z : α

structure RGBValue where
  red : Nat
  green : Nat
  blue : Nat

structure RedGreenPoint (α : Type u) extends Point α, RGBValue where
  no_blue : blue = 0

def p : Point Nat :=
  { x := 10, y := 10, z := 20 }

def rgp : RedGreenPoint Nat :=
  { p with red := 200, green := 40, blue := 0, no_blue := rfl }

example : rgp.x   = 10 := rfl
example : rgp.red = 200 := rfl


class Add (a : Type) where
  add : a → a → a

instance [Add a] : Add (Array a) where
  add x y := Array.zipWith x y (· + ·)

instance [Inhabited a] [Inhabited b] : Inhabited (a × b) where
  default := (default, default)


#print inferInstance

def foo : Inhabited (Nat × Nat) :=
  inferInstance

theorem ex : foo.default = (default, default) :=
  rfl

structure Rational where
  num : Int
  den : Nat
  inv : den ≠ 0

instance : OfNat Rational n where
  ofNat := { num := n, den := 1, inv := by decide }

instance : ToString Rational where
  toString r := s!"{r.num}/{r.den}"

#eval (2 : Rational) -- 2/1

#check (2 : Rational) -- Rational
#check (2 : Nat)      -- Nat

class HMul (α : Type u) (β : Type v) (γ : outParam (Type w)) where
  hMul : α → β → γ

export HMul (hMul)

@[default_instance]
instance : HMul Int Int Int where
  hMul := Int.mul

local instance : Add Point where
  add a b := { x := a.x + b.x, y := a.y + b.y }

attribute [-instance] addPoint

namespace Point

scoped instance : Add Point where
  add a b := { x := a.x + b.x, y := a.y + b.y }

def double (p : Point) :=
  p + p

end Point

open Classical
noncomputable scoped
instance (priority := low) propDecidable (a : Prop) : Decidable a :=
  choice <| match em a with
    | Or.inl h => ⟨isTrue h⟩
    | Or.inr h => ⟨isFalse h⟩
```
