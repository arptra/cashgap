import { apiJSON, bindLiquidityChart, escapeHtml, renderLiquidityChart, setApiKey } from './forecast-widget.js';
const app = document.querySelector('#app');
if (!app) {
    throw new Error('Корневой элемент приложения не найден');
}
const icons = {
    search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
    refresh: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></svg>',
    users: '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><path d="M3 19c.5-4 2.5-6 6-6s5.5 2 6 6M16 5c2 0 3 1 3 3s-1 3-3 3M17 13c2.5.5 3.8 2.5 4 5"/></svg>',
    settings: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19 13.5v-3l-2-.7-.8-1.8.9-2-2.1-2-2 .9-1.9-.8-.6-2h-3l-.7 2-1.8.8-2-.9-2 2.1.9 2-.8 1.9-2 .6v3l2 .7.8 1.8-.9 2L3 20l2-.9 1.9.8.6 2h3l.7-2 1.8-.8 2 .9 2-2.1-.9-2 .8-1.9z"/></svg>',
    bell: '<svg viewBox="0 0 24 24"><path d="M5 17h14l-2-3V9a5 5 0 0 0-10 0v5zM10 21h4"/></svg>',
    logout: '<svg viewBox="0 0 24 24"><path d="M14 5h5v14h-5M10 8l4 4-4 4M14 12H3"/></svg>',
    plus: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>',
    arrowIn: '<svg viewBox="0 0 24 24"><path d="M18 3v7h-7M17 4l-7 7M19 14v5H5V5h5"/></svg>',
    card: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M7 15h4"/></svg>',
    document: '<svg viewBox="0 0 24 24"><path d="M6 3h9l4 4v14H6zM14 3v5h5M9 12h7M9 16h5"/></svg>',
    briefcase: '<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V4h8v3M3 12h18M10 12v2h4v-2"/></svg>',
    chart: '<svg viewBox="0 0 24 24"><path d="M4 20V9M10 20V4M16 20v-7M22 20H2"/></svg>',
    ai: '<svg viewBox="0 0 24 24"><circle cx="9" cy="7" r="3"/><path d="M4 19v-2c0-3 2-5 5-5s5 2 5 5v2M17 5v6M14 8h6M17 15v5M14.5 17.5h5"/></svg>',
    home: '<svg viewBox="0 0 24 24"><path d="m3 11 9-8 9 8v10H7v-8h10v8"/></svg>',
    wallet: '<svg viewBox="0 0 24 24"><path d="M4 6h14a2 2 0 0 1 2 2v11H4a2 2 0 0 1-2-2V7a3 3 0 0 1 3-3h12"/><path d="M15 11h7v5h-7a2.5 2.5 0 0 1 0-5"/></svg>',
    clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="13" r="8"/><path d="M12 9v5l3 2M8 2h8"/></svg>',
    export: '<svg viewBox="0 0 24 24"><path d="M12 16V3M8 7l4-4 4 4M5 12v8h14v-8"/></svg>',
    info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></svg>',
    chevron: '<svg viewBox="0 0 24 24"><path d="m8 14 4-4 4 4"/></svg>',
    panelToggle: '<svg viewBox="0 0 24 24"><path d="M9 7l5 5-5 5"/><path d="M18 4v16"/></svg>',
    paperclip: '<svg viewBox="0 0 24 24"><path d="m8 12 6-6a4 4 0 0 1 6 6L10 22a6 6 0 0 1-8-8L12 4"/></svg>',
    calendar: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M8 2v6M16 2v6M3 11h18"/></svg>'
};
const sidebarItems = [
    ['Создать', icons.plus], ['Платежи', icons.arrowIn], ['Переводы', icons.logout],
    ['Документы', icons.document], ['Контрагенты', icons.briefcase], ['Календарь', icons.calendar],
    ['Счета', icons.card], ['Сервисы', icons.settings], ['Помощники', icons.users],
    ['ИИ адъютант про', icons.ai], ['Аналитика', icons.chart], ['Отчёты', icons.document],
    ['Главная', icons.home], ['Касса', icons.wallet], ['Сотрудники', icons.users],
    ['Экспорт', icons.export], ['Карты', icons.card], ['История', icons.clock]
];
const formatRubles = (value, compact = false) => {
    if (compact) {
        return `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(value)} млн ₽`;
    }
    return new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value) + ' ₽';
};
const renderSidebar = () => `
  <nav class="sidebar" aria-label="Главное меню">
    ${sidebarItems.map(([label, icon], index) => {
    const active = index === 9;
    return `<button class="side-button${active ? ' is-active' : ''}" type="button"
        ${active ? 'aria-current="page" data-dashboard-trigger' : 'disabled aria-disabled="true"'}
        aria-label="${label}" title="${active ? label : `${label} — недоступно в демо`}">
        ${icon}${active ? '<span class="status-dot"></span>' : ''}
      </button>`;
}).join('')}
    <span class="sidebar-spacer"></span>
    <button class="side-button shop-button" type="button" disabled aria-disabled="true" aria-label="Магазин" title="Магазин — недоступно в демо">${icons.briefcase}</button>
  </nav>`;
const renderHeader = (company) => `
  <header class="topbar">
    <a class="brand" href="#dashboard" aria-label="${company.brand}">
      <img class="brand-logo" src="/adjutant/assets/brand-logo.png" width="760" height="192" alt="" />
    </a>
    <div class="search">${icons.search}<span>Поиск</span></div>
    <div class="account-total">${icons.refresh}<span><strong>${formatRubles(company.totalBalance).replace(' ', ' ')}</strong><small>Демо · всего на счетах</small></span></div>
    <div class="profile"><span class="avatar">${company.initials}</span><span><strong>${company.userName}</strong><small>Демонстрационный профиль</small></span></div>
    <div class="top-actions">
      <button aria-label="Сотрудники" disabled>${icons.users}</button>
      <button aria-label="Настройки" disabled>${icons.settings}</button>
      <button aria-label="Уведомления" disabled>${icons.bell}</button>
      <button aria-label="Выйти" disabled>${icons.logout}</button>
    </div>
  </header>`;
const renderMetrics = (metrics) => `
  <section class="metric-grid" aria-label="Ключевые показатели">
    ${metrics.map(metric => `
      <article class="metric-card metric-${metric.tone}" data-demo>
        <div class="metric-label">${metric.label}<span class="tooltip" title="Показатель рассчитан на основе финансовых данных">?</span></div>
        <div class="metric-reading">
          <strong>${metric.value}</strong>
          ${metric.change ? `<span class="trend trend-${metric.changeDirection}">${metric.changeDirection === 'down' ? '▼' : metric.changeDirection === 'up' ? '▲' : ''} ${metric.change}</span>` : ''}
        </div>
        ${metric.note ? `<small>${metric.note}</small>` : ''}
      </article>`).join('')}
  </section>`;
const renderRecommendation = (recommendation) => {
    const steps = [];
    for (let value = recommendation.minAmount; value <= recommendation.maxAmount; value += recommendation.step) {
        steps.push(value);
    }
    const percent = ((recommendation.selectedAmount - recommendation.minAmount) / (recommendation.maxAmount - recommendation.minAmount)) * 100;
    return `
    <section class="card recommendation-card" data-demo>
      <div class="section-heading"><h2>Рекомендации</h2><button class="icon-button" disabled aria-label="Свернуть">${icons.chevron}</button></div>
      <div class="recommendation-layout">
        <div class="recommendation-control">
          <div class="recommendation-note"><span>!</span>${recommendation.message}</div>
          <label for="tranche-range">Сумма транша</label>
          <input id="tranche-range" type="range" min="${recommendation.minAmount}" max="${recommendation.maxAmount}" step="${recommendation.step}" value="${recommendation.selectedAmount}" style="--range-progress:${percent}%" />
          <div class="range-labels">${steps.map(value => `<span>${value >= 1000000 ? `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(value / 1000000)} млн ₽` : `${value / 1000} тыс. ₽`}</span>`).join('')}</div>
          <button class="link-button" type="button" data-toast="Настройки рекомендаций доступны в полной версии">Настроить рекомендации</button>
        </div>
        <div class="tranche-summary">
          <span>Сумма транша</span>
          <strong id="tranche-value">${formatRubles(recommendation.selectedAmount)}</strong>
          <button class="secondary-button" type="button" data-toast="Демонстрация интерфейса: реальный запрос не отправляется">${recommendation.secondaryAction}</button>
          <button class="primary-button" type="button" data-toast="Демонстрация интерфейса: транш не оформляется">${recommendation.primaryAction}</button>
        </div>
      </div>
    </section>`;
};
const renderCounterparties = (counterparties) => `
  <article class="card counterparties-card" data-demo>
    <h2>Дисциплина контрагентов (Топ-5)</h2>
    <div class="counterparty-list">
      ${counterparties.map(counterparty => `
        <div class="counterparty-row">
          <span>${counterparty.name}</span>
          <div class="progress"><i style="width:${counterparty.share}%"></i></div>
          <strong>${formatRubles(counterparty.amount, true)}</strong>
        </div>`).join('')}
    </div>
  </article>`;
const renderComparison = (comparison) => `
  <article class="card comparison-card" data-demo>
    <h2>${comparison.title}</h2>
    <p>${comparison.subtitle}<span class="tooltip" title="Расчёт по сопоставимой выборке">?</span></p>
    <div class="segmented-control" role="group" aria-label="Тип сравнения">
      <button class="selected" type="button">Медиана</button><button type="button">Лучший</button>
    </div>
    <div class="comparison-metrics">
      ${comparison.metrics.map(metric => `
        <div class="comparison-metric">
          <span>${metric.label}</span>
          <div><strong>${metric.value}</strong> <em class="trend-${metric.changeDirection}">${metric.changeDirection === 'down' ? '▼' : '▲'} ${metric.change}</em></div>
          <small>${metric.benchmark}</small>
        </div>`).join('')}
    </div>
  </article>`;
const renderCashFlow = (cashFlow) => {
    const width = 820;
    const height = 272;
    const top = 18;
    const bottom = 42;
    const chartHeight = height - top - bottom;
    const step = 106;
    const startX = 66;
    const y = (value) => top + chartHeight - (value / 120) * chartHeight;
    const balancePoints = cashFlow.points.map((point, index) => `${startX + index * step + 8},${y(point.balance)}`).join(' ');
    return `
    <article class="card cashflow-card" data-demo>
      <div class="section-heading"><h2>Движение средств</h2><span class="year-chip">${cashFlow.year} ${icons.calendar}</span></div>
      <svg class="cashflow-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Приток и отток денежных средств по месяцам">
        ${[0, 20, 40, 60, 80, 100, 120].map(value => `<line x1="32" y1="${y(value)}" x2="780" y2="${y(value)}" class="grid-line"/><text x="812" y="${y(value) + 4}" text-anchor="end" class="axis-text">${value}${value === 120 ? ' млн ₽' : ''}</text>`).join('')}
        ${cashFlow.points.map((point, index) => {
        const x = startX + index * step;
        const inflowHeight = chartHeight - (y(point.inflow) - top);
        const outflowHeight = chartHeight - (y(point.outflow) - top);
        return `<rect x="${x}" y="${y(point.outflow)}" width="12" height="${outflowHeight}" rx="5" class="outflow-bar"><title>Отток: ${point.outflow} млн ₽</title></rect>
            <rect x="${x + 14}" y="${y(point.inflow)}" width="12" height="${inflowHeight}" rx="5" class="inflow-bar"><title>Приток: ${point.inflow} млн ₽</title></rect>
            <text x="${x + 8}" y="${height - 16}" text-anchor="middle" class="axis-text">${point.month}</text>`;
    }).join('')}
        <polyline points="${balancePoints}" class="balance-line"/>
      </svg>
      <div class="legend cashflow-legend"><span><i class="inflow-color"></i>Приток</span><span><i class="outflow-color"></i>Отток</span></div>
    </article>`;
};
const renderAssistant = (assistant) => `
  <aside class="assistant-panel" id="assistant-panel" aria-label="Гига-чат">
    <div class="assistant-header">
      <strong>${assistant.question.slice(0, 27)}…</strong>
      <button class="assistant-toggle" type="button" data-assistant-toggle aria-controls="assistant-panel" aria-expanded="true" aria-label="Свернуть Гига-чат" title="Свернуть Гига-чат">${icons.panelToggle}</button>
      <button disabled aria-label="Открыть отдельно">${icons.export}</button>
      <button disabled aria-label="Справка">${icons.document}</button>
    </div>
    <div class="assistant-scroll">
      <div class="question-bubble">${assistant.question}</div>
      <p>${assistant.introduction}</p>
      <ol>${assistant.sections.map(section => `<li><h3>${section.title}</h3><p>${section.body}</p></li>`).join('')}</ol>
      <div class="assistant-thinking"><span class="assistant-orb"></span>Демо-текст, не анализ выбранного ИНН</div>
    </div>
    <div class="assistant-input"><span>Напишите сообщение</span><button type="button" disabled aria-label="Прикрепить файл">${icons.paperclip}</button></div>
    <small class="assistant-disclaimer">Рекомендуем проверять важное. <u>Подробнее</u></small>
  </aside>`;
const renderDashboard = (data) => {
    app.innerHTML = `
    <div class="application-shell">
      ${renderHeader(data.company)}
      <div class="workspace">
        ${renderSidebar()}
        <main class="dashboard" id="dashboard">
          <div class="dashboard-content">
            <section class="hero">
              <h1>ИИ адъютант про</h1>
              <p>Проактивное управление ликвидностью вашего бизнеса</p>
            </section>
            <div class="info-banner">${icons.info}<span>${escapeHtml(data.demoNotice)}</span></div>
            ${renderMetrics(data.metrics)}
            <button class="ai-question" type="button" data-toast="Гига-чат уже подготовил ответ справа"><span class="assistant-orb"></span>Почему возникает дефицит ликвидности?</button>
            ${renderRecommendation(data.recommendation)}
            <section class="two-column-grid">
              ${renderLiquidityChart(data.liquidityAnalysis)}
              ${renderCounterparties(data.counterparties)}
            </section>
            <section class="two-column-grid comparison-grid">
              ${data.comparisons.map(renderComparison).join('')}
            </section>
            ${renderCashFlow(data.cashFlow)}
            <footer class="dashboard-footer"><span>Прогноз из сохранённого файла · остальные виджеты демонстрационные</span><span>© 2026 СБЕР Бизнес</span></footer>
          </div>
        </main>
        ${renderAssistant(data.assistant)}
      </div>
      <div class="toast" role="status" aria-live="polite"></div>
    </div>`;
    bindInteractions();
    currentLiquidity = data.liquidityAnalysis;
    bindLiquidityChart(currentLiquidity, refreshLiquidity);
};
const showToast = (message) => {
    const toast = document.querySelector('.toast');
    if (!toast)
        return;
    toast.textContent = message;
    toast.classList.add('is-visible');
    window.setTimeout(() => toast.classList.remove('is-visible'), 2400);
};
const setAssistantCollapsed = (collapsed, announce = false) => {
    const shell = document.querySelector('.application-shell');
    const panel = document.querySelector('.assistant-panel');
    const toggle = document.querySelector('[data-assistant-toggle]');
    if (!shell || !panel || !toggle)
        return;
    shell.classList.toggle('assistant-collapsed', collapsed);
    panel.classList.toggle('is-collapsed', collapsed);
    panel.dataset.state = collapsed ? 'collapsed' : 'expanded';
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute('aria-label', collapsed ? 'Развернуть Гига-чат' : 'Свернуть Гига-чат');
    toggle.title = collapsed ? 'Развернуть Гига-чат' : 'Свернуть Гига-чат';
    if (announce) {
        showToast(collapsed ? 'Гига-чат свёрнут в боковую панель' : 'Гига-чат развёрнут');
    }
};
const bindInteractions = () => {
    const range = document.querySelector('#tranche-range');
    const amount = document.querySelector('#tranche-value');
    range?.addEventListener('input', () => {
        const min = Number(range.min);
        const max = Number(range.max);
        const value = Number(range.value);
        range.style.setProperty('--range-progress', `${((value - min) / (max - min)) * 100}%`);
        if (amount)
            amount.textContent = formatRubles(value);
    });
    document.querySelectorAll('[data-toast]').forEach(element => {
        element.addEventListener('click', () => showToast(element.dataset.toast ?? 'Готово'));
    });
    document.querySelector('.banner-close')?.addEventListener('click', event => {
        event.currentTarget.closest('.info-banner')?.remove();
    });
    document.querySelector('[data-dashboard-trigger]')?.addEventListener('click', () => {
        document.querySelector('#dashboard')?.scrollTo({ top: 0, behavior: 'smooth' });
        showToast('Виджет «ИИ адъютант про» открыт');
    });
    const assistantToggle = document.querySelector('[data-assistant-toggle]');
    const savedAssistantState = window.localStorage.getItem('assistant-panel-collapsed');
    const startsCollapsed = savedAssistantState === null
        ? window.matchMedia('(max-width: 1060px)').matches
        : savedAssistantState === 'true';
    setAssistantCollapsed(startsCollapsed);
    assistantToggle?.addEventListener('click', () => {
        const panel = document.querySelector('.assistant-panel');
        const collapsed = !panel?.classList.contains('is-collapsed');
        setAssistantCollapsed(collapsed, true);
        window.localStorage.setItem('assistant-panel-collapsed', String(collapsed));
    });
    document.querySelectorAll('.segmented-control').forEach(control => {
        control.querySelectorAll('button').forEach(button => {
            button.addEventListener('click', () => {
                control.querySelectorAll('button').forEach(item => item.classList.toggle('selected', item === button));
                showToast(`Выбрано сравнение: ${button.textContent?.toLowerCase()}`);
            });
        });
    });
};
let currentLiquidity;
let forecastRequest = 0;
const fetchDashboard = (query = {}) => {
    const params = new URLSearchParams();
    Object.entries(query).forEach(([key, value]) => { if (value !== undefined)
        params.set(key, value); });
    return apiJSON('/api/dashboard?' + params.toString());
};
const refreshLiquidity = async (query) => {
    const request = ++forecastRequest;
    const root = document.querySelector('#liquidity-widget');
    if (!root)
        return;
    root.setAttribute('aria-busy', 'true');
    root.querySelector('#forecast-result').hidden = true;
    root.querySelector('#forecast-status').textContent = 'Читаем прогноз…';
    try {
        const data = await fetchDashboard(query);
        if (request !== forecastRequest)
            return;
        currentLiquidity = data.liquidityAnalysis;
        root.outerHTML = renderLiquidityChart(currentLiquidity);
        bindLiquidityChart(currentLiquidity, refreshLiquidity);
    }
    catch (error) {
        if (request !== forecastRequest)
            return;
        root.outerHTML = renderLiquidityChart(currentLiquidity);
        bindLiquidityChart(currentLiquidity, refreshLiquidity);
        const restored = document.querySelector('#liquidity-widget');
        restored.querySelector('#forecast-result').hidden = true;
        restored.querySelector('#forecast-status').textContent = error instanceof Error ? error.message : 'Ошибка загрузки прогноза.';
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = 'secondary-button';
        retry.textContent = 'Повторить запрос';
        retry.onclick = () => { void refreshLiquidity(query); };
        restored.querySelector('#forecast-status').append(retry);
    }
};
const loadDashboard = async () => {
    try {
        renderDashboard(await fetchDashboard());
    }
    catch (error) {
        app.innerHTML = '<div class="error-state"><span>!</span><h1>Не удалось загрузить виджеты</h1><p id="boot-error"></p><p>Проверьте Python-сервер прогноза. Если он защищён ключом, введите его ниже.</p><form id="boot-connect"><label>Ключ доступа к API <input id="boot-key" type="password" autocomplete="off"></label><button type="submit">Подключиться / повторить</button></form></div>';
        document.querySelector('#boot-error').textContent = error instanceof Error ? error.message : 'Ошибка API';
        document.querySelector('#boot-connect')?.addEventListener('submit', event => {
            event.preventDefault();
            setApiKey(document.querySelector('#boot-key').value);
            void loadDashboard();
        });
    }
};
void loadDashboard();
