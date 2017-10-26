"""Units of measure used in pygeoid"""

from pint import UnitRegistry, set_application_registry

units = UnitRegistry()
set_application_registry(units)

# define Gal
units.define('Gal = 10**-2 * m/sec**2')

# define Eotvos
units.define('Eotvos = 10**-9 * sec**-2 = E')
