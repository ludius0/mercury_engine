# Dokumentace
Program vznikl jako zadání pro zápočtový program pro předmět [NPRG030](https://lukyjanek.github.io/teaching/21-22/NPRG030.html).

<!--ts-->
   * [O programu](#O-programu)
   * [Value třída, a jak ji použít](#Value-třída,-a-jak-ji-použít)
      * [TOC generation with Github Actions](#toc-generation-with-github-actions)
   * [Vytvoření vlastní funkce](#Vytvoření-vlastní-funkce)
      * [Forward funkce](#Forward-funkce)
      * [Backward funkce](#Backward-funkce)
   * [Jak funguje výpočet gradientu](#Jak-funguje-výpočet-gradientu)
     * [Třída Func a ukládání atributu do třídy Value](#Třída-Func-a-ukládání-atributu-do-třídy-Value)
     * [Backward funkce v třídě Value](#Backward-funkce-v-třídě-Value)
   * [Aritmetické operace a další funkce](#Aritmetické-operace-a-další-funkce)
   * [Příklady použití](#Příklady-použití)
   * [Co se nestihlo](#Co-se-nestihlo)
   * [Průběh práce](#Průběh-práce)
   * [Vlastní zhodnocení](#Vlastní-zhodnocení)
<!--te-->

## O programu
Kdykoliv počítáme gradient je zapotřebí si pamatovat pořadí aritmetických operací, a také pro každou funkci musíme znát jejich funkci pro derivaci. Takovéhle operace se rychle stanou nepraktické na počítačích a v posledních letech nacházíme velkou potřebu takových operací pro [zpětnou propagaci](https://cs2.wiki/wiki/Backpropagation) v [neurálních sítích](https://cs.wikipedia.org/wiki/Um%C4%9Bl%C3%A1_neuronov%C3%A1_s%C3%AD%C5%A5) a další....

Program je navržený tento problém vyřešit (aspoň pro skalár) vytvořením vlastní třídy v programovacím jazyku python jménem `Value`.

## Value třída, a jak ji použít
Třída `Value` v syntaxu se zapisuje podobně jako normalní numerické operace. Kdykoliv vytvoříme třídu, musíme do ní vložit jakékoliv číslo (i complexní čísla fungují, ale ne pro složitější operace). Taky třída `Value` už má v sobě předefinované funkce jako třeba [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) nebo `Add (+)`, která už je definováná v `__add__`. Každá taková operace se ukládá do historie třídy a funkcí `Value().backward()` se vypočítá gradient každé třídy, která byla součástí aritmetické historie dané třídy. Výsledek můžeme zobrazit funkcí `Value().grad`.

```
>>> a = Value(-2)
>>> b = Value(3)
>>> c = a * 2 + (a**3).relu() - Value(2).add(b.exp()) + 1
>>> c
>>> Value(-25.085536923187668)
>>> c.backward()
>>> b.grad
>>> -20.085536923187668
```

## Vytvoření vlastní funkce
Aritmetické operace jsou definovaný ve skriptu `op_math.py` a matematické funkce v `nn.py`. Pokud si budeme chtít vytvořit vlastní, tak potřebujeme vytvořit vlastní třídu, která bude podtřídou třídy `Func` z `func_base.py` (+ `setattr_value()`), aby měla všechny potřebné funkce a proměnné pro kombatibilitu s funkcí `Value().backward()`. Následně musíme vytvořit funkci `forward()` a `backward()`, které jsou definované s `staticmethod` pro rychlejší výpočty.

```
class Sigmoid(Func):
    @staticmethod
    def forward(ctx, x):
        out = math.exp(x)/(1 + math.exp(x))
        ctx.saved_values.extend([out])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_values
        grad_input = grad_output * (out * (1 - out))
        return grad_input
setattr_value(Sigmoid)
```

`setattr_value()` uloží naši vlastní třídu jako atribut třídy `Value()`. Náš příklad by se použil třeba takhle:

```
>>> Value(4).sigmoid()
```

### Forward funkce
`forward(ctx, x, *argv)` potřebuje minimálně 2 vstupy. První je `ctx` do kterého uložíme výsledek naší operace (tato operace je součástí třídy `Func`). Druhý je třída sebe (např: `Value(4).sigmoid()`, druhý vstup je číslo 4 nebo `Value(3).funkce(Value(2), Value(math.pi))` má zase tři vstupy kromě `ctx`).
> třída `Func` si ukládá výsledky výpočtu jako list `saved_values`

### Backward funkce
Tuto funkci potřebujeme pro výpočet gradientu každé třídy. Jako první vstup je `ctx` ve kterým máme uložený výsledky z `forward()` (vždy předpokládáme minimálně 2 výstupy z `ctx`). Druhý vstup je gradient z předchozího výpočtu (v Value().backward()) a nakonec vrátíme vypočítaný gradient, která procestuje dalším `backward()` z předchozích operací
> třída `Value` má vlastní `backward()` funkci.

## Jak funguje výpočet gradientu
Každá operační třída s `forward()` a `backward()` je podtřídou třídy `Func`, která ukládá do sebe všechny důležité parametry, které jsou později použity u třídy `Value`, když použijeme vlastní `backward()` funkci (která používá `backward()` operační třídy).

### Třída Func a ukládání atributu do třídy Value
Jak už šlo vidět v sekci "Vytvoření vlastní funkce" nebo v `op_math.py`. Každá výpočetní třída musí být podtřídou třídy `Func` a při použití `setattr_value()` se třída uloží jako atribut třídy `Value`.

```
class Func:
    """
    Backbone of every class used for mathematical computing (in op_math.py).
    During forward() it will store data for later use case of backward().
    """
    def __init__(self, *values, **kwargs):
        self.parents = values
        self.saved_values = []
    
    def later2backward(self, *values):
        self.saved_values.extend(values)

def setattr_value(cls):
    """ 
    'setattr_tensor(cls)' assign child class (like in op_math.py)
    of Func class as in-build function of Value class (value_base.py).

    So for example the class Add (from op_math.py) is inherited in the Value class
    as 'Value().add()' and it use call_func(*args, **kwargs),
    which automatically do:
        1. check if all arguments (*args) are Value class
        2. apply forward() of operational function 'op func' (child class of Func class), 
            this return a new Value class. (Additionaly first input is 'it self' for storing
            data for backward() (computing derivative))
        3. insert op func into the new Value class (in _ctx) 
            for future reference when computing gradient 
            (using backward() of op func for Value().backward())
        4. return the new Value()
    """
    # cls -> operation (op)
    def call_func(*args, **kwargs):
        # check if each arg is Value() (if not then convert to one)
        # so operations like "Value(10) + 5" can work
        parents = tuple(t if isinstance(t, Value) else Value(t) for t in args)
        # computing forward of cls (op func)
        ctx = cls(*parents) # for later2backward()
        ret = Value(cls.forward(ctx, *[t.data for t in parents], **kwargs))
        ret._ctx = ctx # save for backward()
        return ret
    setattr(Value, cls.__name__.lower(), call_func)
```

Hlavní rolí `Func` je si uložit předchozí třídy `Value` a předchozí operační třídy. Při použití každé funkce třídy `Value` vytvoříme vždy novou třídu `Value`, a protože chceme později provést výpočet gradientu, tak si musíme pamatovat celou historii.
> Nejnovější třída `Value` by teoreticky byla schopná vykreslit svoji historii jako rodný list (strom)

Pokaždé když použijeme `setattr_value()` pro operační třídu, která uloží atribut třídy `Value`, tak použitím jména operační třídy (např: `class ReLU(Func)` --> `Value(7).relu()`), tak to automaticky nepoužije `forward()`, ale vlastní funkci `call_func(*args, **kwargs)`, která zkontroluje jestli všechny vstupy jsou třídy `Value`, a pokud není tak z nich udělá. Díky tomu syntax jako tenhle: `Value(10) - 4` bude fungovat stejně jako `Value(10) - Value(4)`. 

Dále to uloží rodiče (vstupy) a `ctx` reprezentuje sebe sama jako vlastní třídu `Value` se kterou právě používáme operační třídu (např: `Value().add()`) s další třídou `Value` (ne nutně, např: `Value(-3).relu()`). Zároveň to použijeme jako úložiště parametrů při `forward()` a `backward()`. Následně vrátíme novou třídu `Value` (`ret`), která je výsledkem naší operační třídy a do té uložíme `ctx` jako úložiště rodičů a výsledky operace. Jako poslední vrátíme `ret`.

### Backward funkce v třídě Value
```
def backward(self, allow_fill=True):
    if self._ctx is None: return # no history
    if self.grad is None and allow_fill: self.grad = 1 # create gradient 1 for first one (one using this func)
    assert (self.grad is not None)
    """"
    Build ordered list of history of all computations of Value classes.
    """
    topo = []   # topological order
    visited = set()
    def build_topo(x):
        visited.add(x)
        if x._ctx is not None:
            for child in x._ctx.parents:
                if child not in visited:
                    build_topo(child)
            topo.append(x)
    build_topo(self)

    for value in reversed(topo):
        """
        Looping back and for each Value() involved in mathematical computing
        to compute its gradient.
        """
        grads = value._ctx.backward(value._ctx, value.grad)
        if not isinstance(grads, tuple): grads = (grads,) # for iteration
        for v, g in zip(value._ctx.parents, grads):
            if g is None:   continue # skip ones without parents
            v.grad = g if v.grad is None else (v.grad + g)
```

Nejdřív si vybudujeme list historie všech tříd `Value` (uložené v `_ctx`, což jsou `ctx` z `setattr_value()`) a jejich operací.

Potom pro každou třídu použijeme funkci `backward()` jejich operační třídy, které byly součástí. Takhle nám gradient docestuje až na úplný konec, což byl začátek (úplně první třídy `Value`). Jenom pro první gradient musíme první definovat jako 1 (pokud není definovaný).

## Aritmetické operace a další funkce
Pro výpočet hlavních aritmetických operací nám stačí základní tři: sčítání, násobek, mocnina. Z nich si jednoduše vyjádříme zbytek: (vše v `op_math.py`) (nelze například pro logaritmus).

```
class Value:
    def __neg__(self):
        return self * -1
    
    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        return self.pow(other)
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
```

Jedinná další implementovaná funkce je v `nn.py`, která je určena pro neurální sítě a to `ReLU()`:

```
class ReLU(Func):
    """
    >>> (Value(-42).relu(), Value(42).relu())
    >>> (Value(0), Value(42))
    """
    @staticmethod
    def forward(ctx, x):
        ctx.saved_values.extend([x])
        return x if x > 0 else 0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_values
        return grad_output * (input >= 0)
setattr_value(ReLU)
```

## Příklady použití
Příklady použití jsou vypsané v [`test.ipynb`](https://jupyter.org/). První příklad je zaměřený na různorodé použítí operačních tříd. Druhý je na implementace vlastní operační třídy. Třetí je použití komplexních čísel.

## Co se nestihlo
Součástí zápočtového programu měla být jednoduchá implementace neurálních sití v `nn.py`, jako další demonstrace využití. Zárověň měl být program navržen pro všechny případy komplexních čísel.

## Průběh práce
Nápad je inspirován knihovnami [pytoch](https://pytorch.org/) a [micrograd](https://github.com/karpathy/micrograd). Původně byl program navržen pro tenzory místo skaláru, ale to by udělalo program zbytečně težší pro něco co má sloužit jako ověření konceptu.
Taky byla i napsaná implementace neurálních sítích, ale měl jsem v návrhu bug, na který jsem po spoustu hodinách nepřišel, a byl jsem nespokojený s kódem, který byl zbytečne komplikovaný, proto jsem se rozhodl vše přepsat a nakonec nezbyl čas pro implementaci neurální sítě.

## Vlastní zhodnocení
Po přepsání jsem s kódem spokojený. Ze začátku se může zdát komplikovaný, ale po krátkém přečtení jde vidět jednoduchost konceptu a modularita programu. Nejsem úplně potěšen z toho, že jsem nestihnul implementaci jednoduchých neurálních sítí, které jsou momentálně primární výpočetní použítí gradientu, ale stále lepší než nestihnou odevzdání programu.