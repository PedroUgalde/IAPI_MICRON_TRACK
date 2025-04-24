
# ========== CONFIGURACIÓN INICIAL ========== #
import pulp
import pandas as pd
import numpy as np

# Cargar datos
file_path = 'Hackaton DB Final 04.21.xlsx'
supply_demand = pd.read_excel(file_path, sheet_name='Supply_Demand')
density_df = pd.read_excel(file_path, sheet_name='Density per Wafer')
capacity_df = pd.read_excel(file_path, sheet_name='Boundary Conditions')
#yield_df = pd.read_excel(file_path, sheet_name='Yield')

# Definir productos y trimestres
products = ['21A', '22B', '23C']
quarters = [col for col in supply_demand.columns if col.startswith(('Q1', 'Q2', 'Q3', 'Q4'))]

# Separar trimestres históricos y futuros
historical_quarters = []
future_quarters = []
for q in quarters:
    if not pd.isna(supply_demand.loc[
        (supply_demand['Product ID'] == '21A') & 
        (supply_demand['Attribute'] == 'Total Projected Inventory Balance'), q
    ].values[0]):
        historical_quarters.append(q)
    else:
        future_quarters.append(q)

# Extraer semanas y porcentajes de capacidad
week_percentages = {
    'Wk1': 0.055, 'Wk2': 0.055, 'Wk3': 0.055, 'Wk4': 0.055, 'Wk5': 0.055,
    'Wk6': 0.055, 'Wk7': 0.055, 'Wk8': 0.055, 'Wk9': 0.06, 'Wk10': 0.10,
    'Wk11': 0.10, 'Wk12': 0.15, 'Wk13': 0.15
}
weeks = list(week_percentages.keys())

# ========== CREACIÓN DEL MODELO ========== #
prob = pulp.LpProblem(
    "Optimizacion_Hibrida_SST", 
    pulp.LpMinimize
    )

# ========== VARIABLES ========== #
production = pulp.LpVariable.dicts(
    "Produccion", 
    [(prod, q) for prod in products for q in quarters], 
    lowBound=0, 
    cat='Integer'
    )

inventory = pulp.LpVariable.dicts(
    "Inventario", 
    [(prod, q) for prod in products for q in quarters], 
    lowBound=0
    )

deviation = pulp.LpVariable.dicts(
    "Desviacion", 
    [(prod, q) for prod in products for q in historical_quarters], 
    lowBound=0
    )

shortfall = pulp.LpVariable.dicts(
    "Shortfall",
    [(prod, q) for prod in products for q in quarters],
    lowBound=0
)

transfer_22B = pulp.LpVariable.dicts(
    "Transfer_22B", 
    quarters, 
    lowBound=0
    )

transfer_23C = pulp.LpVariable.dicts(
    "Transfer_23C", 
    quarters, 
    lowBound=0
    )

# Producción semanal estimada (aproximación a partir de la trimestral)
weekly_prod = pulp.LpVariable.dicts(
    "Prod_Semanal",
    [(prod, q, week) for prod in products for q in quarters for week in weeks],
    lowBound=0
)
ramp_penalties = []
ramp_diff_vars = {}

for prod in products:
    for q in quarters:
        for i in range(len(weeks) - 1):
            wk1 = weeks[i]
            wk2 = weeks[i + 1]
            key = (prod, q, wk1, wk2)
            diff_var = pulp.LpVariable(f"RampDiff_{prod}_{q}_{wk1}_{wk2}", lowBound=0)
            ramp_diff_vars[key] = diff_var

            # Diferencia absoluta entre semanas
            prob += weekly_prod[(prod, q, wk2)] - weekly_prod[(prod, q, wk1)] <= diff_var
            prob += weekly_prod[(prod, q, wk1)] - weekly_prod[(prod, q, wk2)] <= diff_var

            # Penalización por exceso sobre 560
            excess_var = pulp.LpVariable(f"RampPenalty_{prod}_{q}_{wk1}_{wk2}", lowBound=0)
            prob += diff_var - 560 <= excess_var
            ramp_penalties.append(excess_var)

# ========== OBJETIVO ========== #
# ========== OBJETIVO CON PRIORIDADES ========== #
priority_weights = {
    '21A': 1000,
    '22B': 100,
    '23C': 10
}

main_objective = (
    pulp.lpSum(
        deviation[(prod, q)] for prod in products for q in historical_quarters
    )
    + pulp.lpSum(
        inventory[(prod, q)] for prod in products for q in future_quarters
    )
    + pulp.lpSum(
        shortfall[(prod, q)] * priority_weights[prod]
        for prod in products for q in quarters
    )
)

prob += main_objective + 0.01 * pulp.lpSum(ramp_penalties)



# ========== RESTRICCIONES ========== #
density = {
    '21A': 94500, 
    '22B': 69300, 
    '23C': 66850
    }

# Lectura y procesamiento robusto del yield
raw_df = pd.read_excel(file_path, sheet_name='Yield', header=None)

# Extraer trimestres y fechas
quarters_from_yield = raw_df.iloc[0, 2:].values
dates = raw_df.iloc[1, 2:].values

# Extraer los valores por producto
yield_records = []
for i in range(2, raw_df.shape[0]):
    product_id = raw_df.iloc[i, 0]
    values = raw_df.iloc[i, 2:].values

    for j in range(len(values)):
        raw_percent = str(values[j]).replace(',', '.').replace('%', '').strip()
        try:
            percent = float(raw_percent)
        except ValueError:
            percent = None

        yield_records.append({
            "ID": product_id,
            "Q": quarters_from_yield[j],
            "date": dates[j],
            "percent": percent
        })

# Crear DataFrame limpio
yield_df = pd.DataFrame(yield_records)

# Diccionario final: yield_data[producto][quarter] = promedio de percent
yield_data = {
    producto: group.groupby('Q')['percent'].mean().to_dict()
    for producto, group in yield_df.groupby('ID')
}


for col in yield_df.columns:
    if isinstance(col, tuple) and len(col) >= 2:
        quarter = col[0]
        for prod in products:
            try:
                yield_val = yield_df.loc[prod, col]
                yield_data[prod].setdefault(quarter, []).append(yield_val)
            except:
                continue
            
for prod in products:
    for quarter in yield_data[prod]:
        yield_data[prod][quarter] = np.mean(yield_data[prod][quarter]) if yield_data[prod][quarter] else 0.9

# Demanda efectiva por producto y trimestre
# Restricción de demanda efectiva con posibilidad de incumplimiento penalizado
for prod in products:
    for q in quarters:
        try:
            demand = supply_demand[
                (supply_demand['Product ID'] == prod) & 
                (supply_demand['Attribute'] == 'EffectiveDemand')
            ][q].values[0]

            prob += (
                production[(prod, q)] * density[prod] * yield_data[prod].get(q, 0.9)
                + shortfall[(prod, q)]
                >= demand
            )
        except:
            continue


# Inventario total entre 70M y 140M bytes
for q in quarters:
    total_inventory_bytes = pulp.lpSum(inventory[(prod, q)] * density[prod] for prod in products)
    prob += total_inventory_bytes >= 70_000_000
    prob += total_inventory_bytes <= 140_000_000

# Producción en múltiplos de 5
for prod in products:
    for q in quarters:
        int_var = pulp.LpVariable(f"int_div_{prod}_{q}", lowBound=0, cat='Integer')
        prob += (production[(prod, q)] == 5 * int_var)

# Relacionar la producción semanal estimada con la producción trimestral
for prod in products:
    for q in quarters:
        for week in weeks:
            prob += weekly_prod[(prod, q, week)] == production[(prod, q)] * week_percentages[week]


# Capacidad semanal acumulada no puede exceder 100%
total_capacity_by_week = {q: pulp.lpSum(production[(prod, q)] for prod in products) for q in quarters}
for q in quarters:
    for week in weeks:
        pct = week_percentages[week]
        prob += total_capacity_by_week[q] * pct <= total_capacity_by_week[q], f"week_pct_limit_{q}_{week}"

# ===== TRANSFERENCIA DE INVENTARIO SST ENTRE PRODUCTOS =====
for q in quarters:
    # Transferencia de inventario sobrante de 21A a 22B
    prob += transfer_22B[q] <= inventory[('21A', q)]

    # Transferencia de inventario sobrante de 22B a 23C (incluye lo que recibió de 21A)
    prob += transfer_23C[q] <= inventory[('22B', q)] + transfer_22B[q]

    # Ajustar el inventario real después de la transferencia
    prob += inventory[('22B', q)] >= transfer_22B[q]
    prob += inventory[('23C', q)] >= transfer_23C[q]

    # Restar el inventario transferido a 22B y 23C
    prob += inventory[('21A', q)] - transfer_22B[q] >= 0
    prob += inventory[('22B', q)] + transfer_22B[q] - transfer_23C[q] >= 0

# ========== SOLUCIÓN ========== #
result = prob.solve()

# Verificar que los valores son múltiplos de 5
for prod in products:
    for q in quarters:
        val = production[(prod, q)].varValue
        if val % 5 != 0:
            print(f"¡Advertencia! {prod} en {q} no es múltiplo de 5: {val}")

# Resolución tolerante a fallos
result = prob.solve()

# Si no hay solución óptima, remover restricciones de mínimo semanal y reintentar
if pulp.LpStatus[prob.status] != "Optimal":
    print("\n  No se encontró solución óptima. Se eliminan restricciones mínimas semanales y se vuelve a intentar...")
    if 'min_wafer_constraints' in locals():
        for cname in min_wafer_constraints:
            prob.constraints.pop(cname, None)
        result = prob.solve()

if prob.objective is not None:
    print("Valor objetivo (desviaciones + inventario futuro):", pulp.value(prob.objective))
else:
    print(" No se definió la función objetivo o se perdió durante la ejecución.")

# Mostrar estado y valor objetivo
print("\nEstado:", pulp.LpStatus[prob.status])
print("Valor objetivo (desviaciones + inventario futuro):", pulp.value(prob.objective))

# Mostrar producción e inventario óptimos
for prod in products:
    print(f"\nProducto {prod}:")
    for q in quarters:
        print(f"Trimestre {q}:")
        print(f"  Producción óptima = {production[(prod, q)].varValue}")
        print(f"  Inventario óptimo = {inventory[(prod, q)].varValue}")
        if q in historical_quarters:
            print(f"  Desviación ajustada = {deviation[(prod, q)].varValue}")

# Imprimir producción e inventario de forma más compacta
print("\n=== Producción Óptima por Producto y Trimestre ===")

# Crear un diccionario para almacenar los resultados de producción e inventario
produccion_inventario = {}

# Iterar sobre todos los productos y trimestres
for prod in products:
    produccion_inventario[prod] = {}
    for q in quarters:
        prod_value = production[(prod, q)].varValue
        inv_value = inventory[(prod, q)].varValue
        produccion_inventario[prod][q] = {'produccion': prod_value, 'inventario': inv_value}

# Imprimir los resultados de manera compacta
for prod in products:
    print(f"\nProducto {prod}:")
    for q in quarters:
        prod_value = produccion_inventario[prod][q]['produccion']
        inv_value = produccion_inventario[prod][q]['inventario']
        print(f" Trimestre {q}: Producción = {prod_value:.2f} wafers, Inventario = {inv_value:.2f} wafers")

# Validación de distribución semanal por producto
# Crear un diccionario para almacenar la producción semanal por producto
weekly_production_by_product = {prod: {week: 0 for week in weeks} for prod in products}

# Asignar producción semanal a cada semana (asumiendo trimestre = 13 semanas)
for prod in products:
    for q in quarters:
        wafers_total = production[(prod, q)].varValue or 0
        for week in weeks:
            if week in week_percentages:
                ratio = week_percentages[week]  # Acceder con el nombre de la semana ('Wk1', 'Wk2', ...)
                wafers_week = wafers_total * ratio
                weekly_production_by_product[prod][week] += wafers_week
            else:
                print(f"Advertencia: Semana {week} no encontrada en el diccionario week_percentages.")

# Calcular el porcentaje de producción por semana respecto al total por producto
print("\n=== VALIDACIÓN DE DISTRIBUCIÓN SEMANAL POR PRODUCTO ===")

# Iterar sobre cada producto y mostrar su distribución semanal
for prod in products:
    total_prod = sum(weekly_production_by_product[prod].values())
    print(f"\nProducto {prod} - Total producción trimestral: {total_prod:.2f} wafers")

    acumulado = 0
    for week in weeks:
        produced = weekly_production_by_product[prod][week]
        percentage = (produced / total_prod) * 100
        acumulado += percentage
        print(f"{week}: {produced:.2f} wafers ({percentage:.2f}%) | Acumulado: {acumulado:.2f}%")

    # Validación del acumulado de la producción semanal
    if acumulado > 100.0:
        print("\nALERTA: El porcentaje acumulado supera el 100%. Se debe revisar la asignación.")
    elif acumulado < 99.5:
        print("\nALERTA: El total está por debajo del 100%. Podría faltar demanda.")
    else:
        print("\nTodo en orden: La distribución semanal respeta el 100%.")

# Mostrar transferencias entre productos
print("\n=== Transferencias de Inventario SST ===")
for q in quarters:
    print(f"{q}: Transferencia 21A → 22B = {transfer_22B[q].varValue:.2f} wafers, 22B → 23C = {transfer_23C[q].varValue:.2f} wafers")



# ========== MODIFICACIONES PARA MOSTRAR PRODUCCIÓN EN BYTES ========== #

# Primero agregamos una función para calcular wafers a partir de bytes
def calculate_wafers(bytes_production, density, yield_val):
    """Calcula el número de wafers necesarios para una producción en bytes"""
    if density * yield_val == 0:
        return 0
    return bytes_production / (density * yield_val)

# Modificamos la sección de impresión de resultados para mostrar en bytes
print("\n=== Producción Óptima por Producto y Trimestre (en bytes) ===")

# Diccionario para almacenar producción en bytes y wafers
produccion_bytes_wafers = {}

for prod in products:
    produccion_bytes_wafers[prod] = {}
    for q in quarters:
        # Obtener el yield para este trimestre (usamos 0.9 como default si no existe)
        current_yield = yield_data[prod].get(q, 0.9)
        
        # Calcular producción en bytes
        prod_bytes = production[(prod, q)].varValue * density[prod] * current_yield
        
        # Calcular wafers necesarios usando la fórmula proporcionada
        prod_wafers = calculate_wafers(prod_bytes, density[prod], current_yield)
        
        # Guardar ambos valores
        produccion_bytes_wafers[prod][q] = {
            'bytes': prod_bytes,
            'wafers': prod_wafers,
            'inventario_bytes': inventory[(prod, q)].varValue * density[prod]
        }
        
        # Imprimir resultados
        print(f"Producto {prod} - Trimestre {q}:")
        print(f"  Producción = {prod_bytes:,.2f} bytes ({prod_wafers:,.2f} wafers)")
        print(f"  Inventario = {produccion_bytes_wafers[prod][q]['inventario_bytes']:,.2f} bytes")

# ========== CÁLCULO DE PRODUCCIÓN SEMANAL EN WAFERS ========== #

# Crear DataFrame para la hoja Wafer Plan
wafer_plan_data = []

# Asumimos que cada trimestre tiene 13 semanas (como en tus week_percentages)
for prod in products:
    for q in quarters:
        # Obtener producción total en bytes para el trimestre
        total_bytes = produccion_bytes_wafers[prod][q]['bytes']
        
        # Calcular wafers totales para el trimestre
        current_yield = yield_data[prod].get(q, 0.9)
        total_wafers = calculate_wafers(total_bytes, density[prod], current_yield)
        
        # Distribuir según los porcentajes semanales
        for week, pct in week_percentages.items():
            wafer_plan_data.append({
                'Product ID': prod,
                'Quarter': q,
                'Week': week,
                'Wafers': total_wafers * pct
            })

# Crear DataFrame
wafer_plan_df = pd.DataFrame(wafer_plan_data)

# Pivotar para tener el formato de la hoja Wafer Plan (semanas como columnas)
wafer_plan_pivot = wafer_plan_df.pivot_table(
    index=['Product ID', 'Quarter'],
    columns='Week',
    values='Wafers',
    fill_value=0
).reset_index()

# Reordenar columnas para que coincidan con el formato original
week_columns = ['WW_' + str(i+1).zfill(2) for i in range(13)]
wafer_plan_pivot.columns = ['Product ID', 'Quarter'] + week_columns

# ========== GUARDAR EN EXCEL ========== #

# Cargar el archivo Excel existente
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Guardar la hoja Wafer Plan
    wafer_plan_pivot.to_excel(writer, sheet_name='Wafer Plan', index=False)
    
    # Opcional: Guardar otros resultados en una nueva hoja
    pd.DataFrame.from_dict(produccion_bytes_wafers, orient='index').to_excel(
        writer, sheet_name='Resultados Optimización'
    )

print("\n=== Resultados guardados en el archivo Excel ===")
print(f"- Producción semanal por producto guardada en hoja 'Wafer Plan'")
print(f"- Resumen de resultados guardado en hoja 'Resultados Optimización'")

# ========== VALIDACIÓN FINAL ========== #

print("\n=== Validación Final ===")

# Verificar que la suma de wafers por semana coincide con la producción trimestral
for prod in products:
    for q in quarters:
        total_wafers_trim = produccion_bytes_wafers[prod][q]['wafers']
        sum_wafers_semanales = wafer_plan_pivot[
            (wafer_plan_pivot['Product ID'] == prod) & 
            (wafer_plan_pivot['Quarter'] == q)
        ][week_columns].sum().sum()
        
        print(f"Producto {prod} - Trimestre {q}:")
        print(f"  Wafers totales (cálculo): {total_wafers_trim:,.2f}")
        print(f"  Suma wafers semanales: {sum_wafers_semanales:,.2f}")
        print(f"  Diferencia: {abs(total_wafers_trim - sum_wafers_semanales):,.2f}")


# ========== PREPARACIÓN DE DATOS PARA LAS HOJAS EXCEL ========== #

## 1. Hoja "Check" - Resultados detallados y verificaciones
check_data = []

# Agregar producción e inventario por producto y trimestre
for prod in products:
    for q in quarters:
        current_yield = yield_data[prod].get(q, 0.9)
        row = {
            'Product ID': prod,
            'Quarter': q,
            'Production (bytes)': produccion_bytes_wafers[prod][q]['bytes'],
            'Production (wafers)': produccion_bytes_wafers[prod][q]['wafers']/10,
            'Inventory (bytes)': produccion_bytes_wafers[prod][q]['inventario_bytes'],
            'Yield Used': current_yield,
            'Density': density[prod]
        }
        
        # Agregar desviación si es trimestre histórico
        if q in historical_quarters:
            row['Deviation'] = deviation[(prod, q)].varValue if (prod, q) in deviation else 0
        
        check_data.append(row)

check_df = pd.DataFrame(check_data)

## 2. Hoja "Weekly Production" - Producción semanal en bytes y wafers
weekly_data = []

for prod in products:
    for q in quarters:
        current_yield = yield_data[prod].get(q, 0.9)
        total_bytes = produccion_bytes_wafers[prod][q]['bytes']
        total_wafers = produccion_bytes_wafers[prod][q]['wafers']
        
        for week, pct in week_percentages.items():
            weekly_bytes = total_bytes * pct
            weekly_wafers = total_wafers * pct
            
            weekly_data.append({
                'Product ID': prod,
                'Quarter': q,
                'Week': week,
                'Weekly Percentage': pct,
                'Production (bytes)': weekly_bytes,
                'Production (wafers)': weekly_wafers
            })

weekly_df = pd.DataFrame(weekly_data)

## 3. Hoja "SST Results" - Resultados de Safety Stock Target
sst_data = []

for prod in products:
    for q in quarters:
        try:
            sst_value = supply_demand[
                (supply_demand['Product ID'] == prod) & 
                (supply_demand['Attribute'] == 'Safety Stock Target')
            ][q].values[0]
        except:
            sst_value = 0
            
        sst_data.append({
            'Product ID': prod,
            'Quarter': q,
            'Safety Stock Target (bytes)': sst_value,
            'Safety Stock Target (wafers)': calculate_wafers(sst_value, density[prod], yield_data[prod].get(q, 0.9))
        })

sst_df = pd.DataFrame(sst_data)

## 4. Hoja "Validations" - Resultados de las validaciones
validation_data = []

# Validación de distribución semanal
for prod in products:
    for q in quarters:
        total_wafers_trim = produccion_bytes_wafers[prod][q]['wafers']
        sum_wafers_semanales = wafer_plan_pivot[
            (wafer_plan_pivot['Product ID'] == prod) & 
            (wafer_plan_pivot['Quarter'] == q)
        ][week_columns].sum().sum()
        
        validation_data.append({
            'Validation': 'Weekly Distribution',
            'Product ID': prod,
            'Quarter': q,
            'Total Wafers (calculated)': total_wafers_trim,
            'Sum Weekly Wafers': sum_wafers_semanales,
            'Difference': abs(total_wafers_trim - sum_wafers_semanales),
            'Status': 'OK' if abs(total_wafers_trim - sum_wafers_semanales) < 0.01 else 'Warning'
        })

# Validación de múltiplos de 5
for prod in products:
    for q in quarters:
        val = production[(prod, q)].varValue
        validation_data.append({
            'Validation': 'Multiple of 5',
            'Product ID': prod,
            'Quarter': q,
            'Value': val,
            'Status': 'OK' if val % 5 == 0 else 'Warning'
        })

validation_df = pd.DataFrame(validation_data)

# ========== GUARDADO EN EXCEL ========== #
try:
    # Intentar guardar en el archivo existente
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Hoja con plan de wafers original
        wafer_plan_pivot.to_excel(writer, sheet_name='Wafer Plan', index=False)
        
        # Nuevas hojas con resultados
        check_df.to_excel(writer, sheet_name='Check', index=False)
        weekly_df.to_excel(writer, sheet_name='Weekly Production', index=False)
        sst_df.to_excel(writer, sheet_name='SST Results', index=False)
        validation_df.to_excel(writer, sheet_name='Validations', index=False)
        
        # Hoja resumen con parámetros clave
        pd.DataFrame({
            'Parameter': ['Total Objective Value', 'Status', 'Historical Quarters', 'Future Quarters'],
            'Value': [
                pulp.value(prob.objective),
                pulp.LpStatus[prob.status],
                len(historical_quarters),
                len(future_quarters)
            ]
        }).to_excel(writer, sheet_name='Summary', index=False)
    
    print("\nResultados guardados exitosamente en el archivo Excel con múltiples hojas.")
    
except PermissionError:
    print("\nNo se pudo guardar en el archivo original (puede estar abierto). Creando copia...")
    
    # Guardar en un nuevo archivo si no se puede modificar el original
    new_file_path = file_path.replace('.xlsx', '_OPTIMIZED.xlsx')
    
    with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
        wafer_plan_pivot.to_excel(writer, sheet_name='Wafer Plan', index=False)
        check_df.to_excel(writer, sheet_name='Check', index=False)
        weekly_df.to_excel(writer, sheet_name='Weekly Production', index=False)
        sst_df.to_excel(writer, sheet_name='SST Results', index=False)
        validation_df.to_excel(writer, sheet_name='Validations', index=False)
        pd.DataFrame({
            'Parameter': ['Total Objective Value', 'Status', 'Historical Quarters', 'Future Quarters'],
            'Value': [
                pulp.value(prob.objective),
                pulp.LpStatus[prob.status],
                len(historical_quarters),
                len(future_quarters)
            ]
        }).to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Resultados guardados en nuevo archivo: {new_file_path}")

print("\nEstructura del archivo Excel generado:")
print("- Wafer Plan: Producción semanal en wafers (formato original)")
print("- Check: Resultados detallados por producto y trimestre")
print("- Weekly Production: Producción semanal en bytes y wafers")
print("- SST Results: Valores de Safety Stock Target")
print("- Validations: Resultados de las validaciones realizadas")
print("- Summary: Resumen ejecutivo de la optimización")