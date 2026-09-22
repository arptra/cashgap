export interface ForecastRow {
  date: string; source_period: string; inflow: number; outflow: number;
  net_flow: number; cumulative_net_flow: number; closing_balance: number | null;
}
export interface LiquidityAnalysis {
  series: { id: 'inflow' | 'outflow' | 'net' | 'balance'; label: string; color: string }[];
  categories: { label: string; inflow: number; outflow: number; net: number; balance: number | null }[];
  rows: ForecastRow[];
  totals: { inflow: number; outflow: number; net_flow: number };
  context: {
    inn: string; period: string; startDate: string | null; endDate: string | null;
    availablePeriods: string[]; availableDates: string[]; dailyAllocationEnabled: boolean;
    openingBalance: number | null; closingBalance: number | null; firstNegativeDate: string | null;
    modelName: string; historyEnd: string | null; source: string; warning: string;
  };
}
export type ForecastQuery = { inn?: string; period?: string; start_date?: string; opening_balance?: string };
let apiKey = '';
let searchSequence = 0;
export const setApiKey = (value: string): void => { apiKey = value.trim(); };
export const escapeHtml = (value: unknown): string => String(value ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] ?? c));
const number = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 });
const money = (value: number): string => number.format(value) + ' ₽';
const dateLabel = (value: string): string => value.length === 7
  ? new Intl.DateTimeFormat('ru-RU', { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(new Date(value + '-01T00:00:00Z'))
  : new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', timeZone: 'UTC' }).format(new Date(value + 'T00:00:00Z'));

export async function apiJSON<T>(path: string): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, { cache: 'no-store', signal: controller.signal,
      headers: { Accept: 'application/json', ...(apiKey ? { 'X-API-Key': apiKey } : {}) } });
    const data = await response.json();
    if (!response.ok) throw new Error(response.status === 401 ? 'Введите корректный ключ доступа к API.' :
      typeof data.detail === 'string' ? data.detail : `Ошибка API: ${response.status}`);
    return data as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('Сервер не ответил за 20 секунд. Повторите запрос.');
    throw error;
  } finally { window.clearTimeout(timer); }
}

export function renderLiquidityChart(data: LiquidityAnalysis): string {
  const ctx = data.context;
  const range = ctx.startDate && ctx.endDate ? `${dateLabel(ctx.startDate)} — ${dateLabel(ctx.endDate)}` : dateLabel(ctx.period);
  const options = (values: string[], current: string | null): string => values.map(value =>
    `<option value="${escapeHtml(value)}"${current === value ? ' selected' : ''}>${escapeHtml(dateLabel(value))}</option>`).join('');
  return `<article class="card chart-card liquidity-card forecast-widget" id="liquidity-widget" aria-busy="false">
    <div class="section-heading"><h2>Анализ ликвидности</h2><span class="forecast-live">Данные прогноза</span></div>
    <p class="forecast-caption">Денежные потоки, не коэффициент ликвидности</p>
    <form class="forecast-controls" id="forecast-controls">
      <label>Поиск ИНН<input id="forecast-search" type="search" placeholder="Первые цифры ИНН" maxlength="64" autocomplete="off"></label>
      <label>Клиент из Parquet<select id="forecast-inn"><option value="${escapeHtml(ctx.inn)}">${escapeHtml(ctx.inn)}</option></select></label>
      <label>Месяц<select id="forecast-period">${options(ctx.availablePeriods, ctx.period)}</select></label>
      ${ctx.dailyAllocationEnabled ? `<label>Начальная дата<select id="forecast-date">${options(ctx.availableDates, ctx.startDate)}</select></label>` : ''}
      <label class="forecast-balance-label">Ваш остаток на начало ${escapeHtml(dateLabel(ctx.startDate ?? ctx.period))}, ₽
        <input id="forecast-balance" inputmode="decimal" placeholder="Необязательно" value="${ctx.openingBalance === null ? '' : escapeHtml(number.format(ctx.openingBalance))}"></label>
      <button class="secondary-button forecast-submit" type="submit">Рассчитать остаток</button>
    </form>
    <p class="forecast-search-status" id="forecast-search-status" role="status"></p>
    <details class="forecast-access"><summary>Ключ доступа к API</summary><label>Ключ<input id="forecast-key" type="password" autocomplete="off"></label><button id="forecast-connect" type="button" class="secondary-button">Подключиться</button></details>
    <p class="forecast-status" id="forecast-status" role="status" aria-live="polite"></p>
    <div class="forecast-result" id="forecast-result">
      <div class="forecast-period-title"><strong>ИНН ${escapeHtml(ctx.inn)}</strong><span>${escapeHtml(range)}</span></div>
      <div class="forecast-totals">
        <div><span>Поступления</span><strong>${money(data.totals.inflow)}</strong></div>
        <div><span>Списания</span><strong>${money(data.totals.outflow)}</strong></div>
        <div><span>Чистый поток</span><strong class="${data.totals.net_flow < 0 ? 'forecast-negative' : ''}">${money(data.totals.net_flow)}</strong></div>
        <div><span>Остаток на конец</span><strong class="${ctx.closingBalance !== null && ctx.closingBalance < 0 ? 'forecast-negative' : ''}">${ctx.closingBalance === null ? 'Не задан' : money(ctx.closingBalance)}</strong></div>
      </div>
      <div class="forecast-chart-scroll">${renderChart(data)}</div>
      <div class="legend forecast-legend">${data.series.map(series => `<span><i class="series-${series.id}"></i>${escapeHtml(series.label)}</span>`).join('')}</div>
      <p class="forecast-method">${escapeHtml(ctx.warning)}</p>
      <p class="forecast-balance-note">${ctx.openingBalance === null ? 'Без начального остатка нельзя оценить достаточность денег.' : ctx.firstNegativeDate
        ? `В этом сценарии отрицательный остаток на конец периода впервые появляется ${escapeHtml(dateLabel(ctx.firstNegativeDate))}. Это не предсказанная дата кассового разрыва.`
        : 'В этом сценарии остатки на конец показанных периодов неотрицательны. Это не гарантия отсутствия кассового разрыва.'} Реальные сроки и порядок платежей неизвестны.</p>
      <details class="forecast-table"><summary>Суммы по ${ctx.dailyAllocationEnabled ? 'дням' : 'месяцу'} · ${data.rows.length} ${ctx.dailyAllocationEnabled ? 'дней' : 'месяц'}</summary>
        <div class="forecast-table-scroll"><table><thead><tr><th>Дата</th><th>Поступления, ₽</th><th>Списания, ₽</th><th>Чистый поток, ₽</th><th>Остаток, ₽</th></tr></thead><tbody>
        ${data.rows.map(row => `<tr><td>${escapeHtml(dateLabel(row.date))}</td><td>${number.format(row.inflow)}</td><td>${number.format(row.outflow)}</td><td>${number.format(row.net_flow)}</td><td>${row.closing_balance === null ? '—' : number.format(row.closing_balance)}</td></tr>`).join('')}
        </tbody></table></div>
      </details>
      <p class="forecast-source">${escapeHtml(ctx.modelName)} · История по: ${ctx.historyEnd ? escapeHtml(dateLabel(ctx.historyEnd)) : 'не указана'} · ${escapeHtml(ctx.source)}. Сохранённый прогноз, без запуска .pt при запросе. Вероятности не рассчитываются.</p>
    </div>
  </article>`;
}

function renderChart(data: LiquidityAnalysis): string {
  const width = Math.max(440, data.categories.length * 55 + 75), height = 240, left = 66, top = 25, bottom = 36;
  const values = data.categories.flatMap(row => data.series.map(series => row[series.id] ?? 0));
  const low = Math.min(0, ...values), high = Math.max(1, ...values), span = high - low;
  const scale = Math.max(Math.abs(low), high) >= 1e6 ? 1e6 : Math.max(Math.abs(low), high) >= 1e3 ? 1e3 : 1;
  const unit = scale === 1e6 ? 'млн ₽' : scale === 1e3 ? 'тыс. ₽' : '₽';
  const y = (value: number): number => top + (high - value) / span * (height - top - bottom);
  const zero = y(0), step = (width - left - 15) / data.categories.length;
  const axis = Array.from({ length: 5 }, (_, i) => {
    const value = low + span * i / 4;
    return `<line x1="${left}" x2="${width - 15}" y1="${y(value)}" y2="${y(value)}" class="forecast-grid"/><text x="${left - 8}" y="${y(value) + 4}" text-anchor="end">${number.format(value / scale)}</text>`;
  }).join('');
  const bars = data.categories.map((row, i) => {
    const x = left + step * (i + .5);
    return data.series.slice(0, 2).map((series, j) => {
      const value = row[series.id] ?? 0;
      return `<rect x="${x + (j ? 2 : -15)}" y="${Math.min(y(value), zero)}" width="12" height="${Math.abs(y(value) - zero)}" rx="3" class="series-${series.id}"><title>${escapeHtml(dateLabel(row.label))} · ${escapeHtml(series.label)}: ${money(value)}</title></rect>`;
    }).join('') + `<text x="${x}" y="${height - 12}" text-anchor="middle">${escapeHtml(row.label.length === 10 ? dateLabel(row.label).slice(0, 5) : row.label)}</text>`;
  }).join('');
  const lineSeries = data.series[2];
  const points = data.categories.map((row, i) => `${left + step * (i + .5)},${y(lineSeries ? row[lineSeries.id] ?? 0 : 0)}`).join(' ');
  const dots = data.categories.map((row, i) => {
    const value = lineSeries ? row[lineSeries.id] ?? 0 : 0;
    return `<circle cx="${left + step * (i + .5)}" cy="${y(value)}" r="3" class="forecast-point"><title>${escapeHtml(dateLabel(row.label))} · ${escapeHtml(lineSeries?.label)}: ${money(value)}</title></circle>`;
  }).join('');
  return `<svg class="forecast-chart" viewBox="0 0 ${width} ${height}" width="${width}" role="img" aria-label="Поступления и списания, ${escapeHtml(lineSeries?.label)}, ${unit}"><text x="${left}" y="14">${unit}</text>${axis}<line x1="${left}" x2="${width - 15}" y1="${zero}" y2="${zero}" class="forecast-zero"/>${bars}<polyline points="${points}" class="forecast-line"/>${dots}</svg>`;
}

export function bindLiquidityChart(data: LiquidityAnalysis, refresh: (query: ForecastQuery) => Promise<void>): void {
  const root = document.querySelector<HTMLElement>('#liquidity-widget');
  if (!root) return;
  const input = <T extends HTMLInputElement | HTMLSelectElement>(id: string): T => root.querySelector<T>('#' + id)!;
  const context = data.context;
  const query = (): ForecastQuery => ({ inn: input<HTMLSelectElement>('forecast-inn').value,
    period: input<HTMLSelectElement>('forecast-period').value,
    ...(context.dailyAllocationEnabled ? { start_date: input<HTMLSelectElement>('forecast-date').value } : {}) });
  let timer = 0;
  const search = input<HTMLInputElement>('forecast-search');
  const selector = input<HTMLSelectElement>('forecast-inn');
  const status = root.querySelector<HTMLElement>('#forecast-search-status')!;
  search.addEventListener('input', () => {
    window.clearTimeout(timer); const seq = ++searchSequence;
    selector.disabled = true;status.textContent = 'Ищем ИНН…';
    timer = window.setTimeout(async () => {
      try {
        const result = await apiJSON<{ inns: string[] }>('/ui/clients?q=' + encodeURIComponent(search.value.trim()));
        if (seq !== searchSequence || !root.isConnected) return;
        selector.replaceChildren(...result.inns.map(inn => new Option(inn, inn)));
        selector.disabled = !result.inns.length;
        if (result.inns.includes(context.inn)) selector.value = context.inn;
        status.textContent = result.inns.length ? 'До 15 совпадений. Выберите ИНН из списка.' : 'Совпадений нет. Измените поиск.';
        // Even a single matching INN requires a deliberate user choice.
        if (!result.inns.includes(context.inn) && result.inns.length) selector.prepend(new Option('Выберите ИНН', '', true, true));
      } catch (error) { if (seq === searchSequence && root.isConnected) status.textContent = error instanceof Error ? error.message : 'Ошибка поиска.'; }
    }, 250);
  });
  selector.addEventListener('change', () => { if (selector.value) void refresh({ inn: selector.value }); });
  input<HTMLSelectElement>('forecast-period').addEventListener('change', () => { void refresh({ inn: context.inn, period: input<HTMLSelectElement>('forecast-period').value }); });
  root.querySelector('#forecast-date')?.addEventListener('change', () => { void refresh({ ...query() }); });
  root.querySelector('form')?.addEventListener('submit', event => {
    event.preventDefault();
    const value = input<HTMLInputElement>('forecast-balance').value.replace(/[\s\u00a0\u202f]/g, '').replace(',', '.');
    if (value && (!/^[+-]?\d+(\.\d{0,2})?$/.test(value) || Math.abs(Number(value)) > 1e12)) {
      root.querySelector<HTMLElement>('#forecast-status')!.textContent = 'Введите остаток в рублях, до двух знаков после запятой и не больше 1 трлн ₽ по модулю.';return;
    }
    if (!selector.value || selector.disabled) { status.textContent = 'Сначала выберите ИНН.';return; }
    void refresh({ ...query(), ...(value ? { opening_balance: value } : {}) });
  });
  input<HTMLInputElement>('forecast-key').value = apiKey;
  root.querySelector('#forecast-connect')?.addEventListener('click', () => {
    setApiKey(input<HTMLInputElement>('forecast-key').value);void refresh({ inn: context.inn, period: context.period });
  });
}
