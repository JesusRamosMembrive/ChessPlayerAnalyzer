# Playbook condensado para agentes de programación (Claude Code / Codex)

> **Propósito**: Este documento unifica **Planificación incremental**, **Arquitectura minimalista**, **Estrategia de tests acumulativa** y **Guía de simplificación de código**. Pégalo como contexto (system/prompt) o mantenlo como referencia operativa para agentes de IA y humanos.
>
> Bases: planificación incremental por **vertical slices** fileciteturn0file1, arquitectura **minimalista con YAGNI** fileciteturn0file2, **test único acumulativo** por fases fileciteturn0file0 y **ronda de simplificación** con criterios objetivos fileciteturn0file3.

---

## 0) Flujo “de 0 a PR” (resumen operativo)

1) **Antes de codificar**  
   - Define un **slice vertical** que entregue valor real en 1–3 días (nada de “infra-only”). Usa el template de fases. fileciteturn0file1  
   - Prefiere la **solución más simple que funcione**; cuestiona cada capa/abstracción. fileciteturn0file2

2) **Implementación**  
   - **Funciones > clases** cuando sea posible; composición > herencia; evitar configuraciones innecesarias. fileciteturn0file2  
   - Construye lo **mínimo necesario hoy** (YAGNI). fileciteturn0file2

3) **Pruebas end‑to‑end**  
   - Actualiza **un solo script** `tests/test_full_flow.sh` con una **nueva Fase N** (acumulativa, con `set -e`, logs claros y cleanup). fileciteturn0file0

4) **Simplificación**  
   - Ejecuta la **“Code Simplification Prompt”** y aplica los cambios manteniendo la misma funcionalidad. fileciteturn0file3

5) **Entrega**  
   - PR pequeño, legible, pasando el full flow; deja apuntes de **qué NO hiciste en esta fase**. fileciteturn0file1

---

## 1) Planificación Incremental (vertical slices)

**Principios clave**  
- **Cada fase entrega funcionalidad end‑to‑end** utilizable por el usuario.  
- **Sencillez primero**: el **MVP del MVP** en 1–3 días, muchas releases pequeñas.  
- **Sin infraestructura prematura**; la arquitectura emerge y se refactoriza con cada fase. fileciteturn0file1

**Preguntas guía por fase**  
1. ¿Cuál es **lo más pequeño** que ya da valor?  
2. ¿El usuario **puede usarlo** al terminar esta fase?  
3. ¿Estoy construyendo **infra o features**? (prefiere features)  
4. ¿Cuál es **la forma más “tonta”** de hacerlo funcionar?  
5. ¿Qué **no** haremos **explícitamente** en esta fase? fileciteturn0file1

**Antipatrón** a evitar: plan basado en capas/infra (repos, servicios, auth, frontend, …).  
**Patrón recomendado**: UI con datos hardcodeados → persistencia simple → auth básica → refactor de data layer → features avanzadas guiadas por feedback. fileciteturn0file1

**Template de Fase**  
```text
Phase N: [Nombre descriptivo]
Goal: [1 frase: valor usuario]
Duration: [1–3 días]
Deliverable: [Feature operativa]
Implementation: [Solución más simple posible]
Notes: [Qué NO hacemos en esta fase]
```  
fileciteturn0file1

---

## 2) Arquitectura minimalista (pragmática)

**Principios**  
- **Simplicidad primero**; evitar over‑engineering y patrones innecesarios.  
- **Mínimas capas**; cuestiona cada clase/interfaz/capa: “¿se necesita de verdad?”.  
- **Pragmático**: **funciones > clases**, **composición > herencia**, **explícito > implícito**.  
- **YAGNI**: construye sólo lo necesario **ahora**. fileciteturn0file2

**Guías de implementación**  
- Nombres simples, funciones pequeñas, estructuras de datos planas y flujo lineal.  
- Camino feliz obvio, usar la stdlib antes que dependencias.  
- Evitar ABCs, factories/builders complejos, middlewares/decorators innecesarios. fileciteturn0file2

**Estructura preferida (pequeño/mediano)**  
```text
project/
├── main.py         # entry point
├── core.py         # lógica principal
├── utils.py        # helpers
├── config.py       # opcional
└── tests/
```  
Evita jerarquías profundas y capas “arquitectónicas” hasta que haya **múltiples implementaciones** que lo justifiquen. fileciteturn0file2

**Plantilla de respuesta técnica (para agentes)**  
```text
Enfoque simple que cumple requisitos:
[Solución directa]
Por qué es suficiente:
- resuelve la necesidad sin complejidad extra
- fácil de entender/modificar
- puntos claros de expansión futura: [X, Y]
Si el alcance crece: considerar [mejoras concretas].
```  
fileciteturn0file2

---

## 3) Estrategia de tests: **script único acumulativo**

**Regla de oro**: un **solo** `tests/test_full_flow.sh` que valida todo el sistema **por fases** (Phase 1, 2, 3, …), con `set -e`, logs descriptivos y **cleanup** final. fileciteturn0file0

**Buenas prácticas**  
- Organizar por **fases/features** con timestamps; mantener cada sección **enfocada**.  
- **Salir al primer fallo** (`set -e`) para localizar rápido.  
- **No** mezclar unit con integration; **no** depender de artefactos previos; **no** pasar de 10–15 min. fileciteturn0file0

**Cuándo dividir** en scripts por fase: sólo si el full flow supera 10–15 min, módulos realmente independientes, entornos distintos o performance testing. fileciteturn0file0

**Prompt breve para agentes al escribir tests**  
- **Añade/actualiza** `test_full_flow.sh` (no crees archivos nuevos).  
- Crea una **nueva sección Fase N** con headers claros.  
- **Construye sobre fases previas** y valida el flujo completo.  
- **Logs descriptivos** y **cleanup** de datos de prueba. fileciteturn0file0

**Skeleton recomendado**  
```bash
#!/bin/bash
set -e
echo "🧪 Starting Full Flow Test..."

# =============================================================================
# PHASE 1: [Setup básico] (implemented: YYYY-MM-DD)
# =============================================================================
echo "📋 Testing Phase 1: [Setup básico]..."
# [comandos de prueba]
echo "✅ Phase 1 passed"

# =============================================================================
# PHASE 2: [Persistencia] (implemented: YYYY-MM-DD)
# =============================================================================
echo "📋 Testing Phase 2: [Persistencia]..."
# [comandos de prueba]
echo "✅ Phase 2 passed"

# =============================================================================
# PHASE 3: [Autenticación] (implemented: YYYY-MM-DD)
# =============================================================================
echo "📋 Testing Phase 3: [Autenticación]..."
# [comandos de prueba]
echo "✅ Phase 3 passed"

# =============================================================================
# CLEANUP
# =============================================================================
echo "🧹 Cleaning up test data..."
# [cleanup]
echo "🎉 All tests passed! System working correctly."
```  
fileciteturn0file0

**Plantilla de sección Fase (dentro del script)**  
```bash
# =============================================================================
# PHASE X: [Feature] (implemented: YYYY-MM-DD)
# =============================================================================
echo "📋 Testing Phase X: [Feature]..."
# 1) setup
# 2) pruebas de la nueva capacidad
# 3) integración con fases previas
# 4) validación end-to-end
echo "✅ Phase X passed"
```  
fileciteturn0file0

---

## 4) Simplificación de código (post‑feature)

**Objetivo**: **misma funcionalidad**, menor complejidad, mayor legibilidad y mantenibilidad. Aplica **el prompt de simplificación** y ejecuta refactors locales (no cosméticos). fileciteturn0file3

**Criterios de simplificación** (resumen)  
1) **Quitar abstracciones prematuras**: interfaces/ABCs/patrones sin valor → funciones simples.  
2) **Reducir dependencias**: eliminar libs poco usadas; preferir soluciones nativas.  
3) **Simplificar estructura**: menos archivos/config si no aportan; estructuras de datos simples.  
4) **Mejorar legibilidad**: nombres claros, menos nesting, condicionales sencillos.  
5) **Aplicar YAGNI**: eliminar configurabilidad y código para “futuros” no requeridos. fileciteturn0file3

**No hacer**: quitar validaciones importantes, romper manejo de errores o sacrificar claridad por brevedad.  
**Formato de respuesta esperado**: cambios propuestos, por qué simplifican, confirmación de funcionalidad intacta y **código resultante**. fileciteturn0file3

**Prompt listo para pegar**  
```text
Please review all the code we just developed and simplify it following these criteria:
[...objetivos y criterios de la guía...]
RESPONSE FORMAT:
1) Cambios principales
2) Justificación por cada cambio
3) Confirmación de funcionalidad intacta
4) Código simplificado
```  
Usa variantes para **config pesada**, **muchas abstracciones** o **muchas dependencias** cuando aplique. fileciteturn0file3

---

## 5) Checklists rápidas

**Antes de implementar**  
- Slice vertical con valor real y alcance ≤ 3 días.  
- Decisiones guiadas por “¿Qué es lo más simple que funcionaría?” y **¿puedo quitar una capa?** fileciteturn0file1turn0file2

**Antes de abrir PR**  
- `test_full_flow.sh` actualizado, pasa en limpio y en < 15 min.  
- Notas de **lo que NO se hizo** en esta fase. fileciteturn0file0turn0file1

**Después del PR (o antes si es pequeño)**  
- Ejecuta la **ronda de simplificación** según criterios. fileciteturn0file3

---

## 6) Prompts breves para agentes

**a) Al iniciar una tarea**  
```text
Actúa como Arquitecto minimalista + Planificador incremental.
Propón 3–5 fases (vertical slices), cada una utilizable.
Para la Fase 1, dame la solución más simple posible (funciones > clases, 0 infra prematura, YAGNI).
Incluye: qué NO harás en Fase 1 y cómo validarás en test_full_flow.sh.
```  
fileciteturn0file1turn0file2

**b) Al escribir/actualizar tests**  
```text
Actualiza tests/test_full_flow.sh agregando "PHASE N: [Feature]".
Usa set -e, logs descriptivos y cleanup.
Valida integración con fases previas y el flujo E2E completo.
```  
fileciteturn0file0

**c) Al cerrar la fase**  
```text
Ejecuta la "Code Simplification Prompt".
Propón refactors que reduzcan complejidad sin cambiar comportamiento.
Entrega difs concisos y justificados.
```  
fileciteturn0file3

---

## 7) Notas finales

- Cada fase **produce software funcionando**, no “fundaciones” teóricas.  
- La **arquitectura emerge** de requisitos reales; se refactoriza con criterio.  
- Las pruebas **protegen el flujo completo**; el script único cuenta la historia del sistema.  
- La simplificación es **parte del proceso**, no un lujo.  
fileciteturn0file1turn0file2turn0file0turn0file3
