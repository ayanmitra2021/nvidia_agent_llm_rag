(async () => {
  const config = __COURSE_LANGUAGE_CONFIG__;
  const labels = config.labels;
  let current = {language: config.language};
  const text = key => labels[current.language][key];
  const translate = message => {const key=Object.keys(labels.en).find(key=>labels.en[key]===message);return key?text(key):message;};
  let host = document.getElementById(config.id);
  // Separate HTML and JavaScript outputs can mount in different render frames.
  for (let frame = 0; !host && frame < 120; frame++) {
    await new Promise(resolve => requestAnimationFrame(resolve));
    host = document.getElementById(config.id);
  }
  if (!host) return;
  const app = window.jupyterapp;
  if (!app) { host.textContent = text('session'); return; }
  await app.restored;
  const widget = Array.from(app.shell.widgets('main')).find(item => item.node.contains(host) && item.context);
  const entrypoint = widget && config.entrypoints.find(name => widget.context.path === name || widget.context.path.endsWith('/' + name));
  if (!entrypoint) {
    host.textContent = text('intro'); return;
  }
  const course = widget.context.path.slice(0, -entrypoint.length).replace(/\/$/, '');
  const base = app.serviceManager.serverSettings.baseUrl;
  const endpoint = base + 'api/course/language';
  const states = window.__dliCourseLanguageStates ||= {};
  const shared = states[course] ||= {busy: false, message: ''};
  host.replaceChildren();
  host.classList.add('dli-language');
  const style = document.createElement('style');
  style.textContent = '.dli-language{color:var(--jp-ui-font-color1,inherit);font:var(--jp-ui-font-size1,14px) var(--jp-ui-font-family,inherit);max-width:640px}.dli-language select{font:inherit;color:inherit;background:var(--jp-layout-color0,Canvas);border:1px solid var(--jp-border-color1,#888);border-radius:2px;padding:4px 24px 4px 8px;margin:0 8px}.dli-language select:focus-visible{outline:2px solid #76b900;outline-offset:2px}.dli-language details{margin-top:8px}.dli-language summary{cursor:pointer}.dli-language p:empty{display:none}.dli-language a{color:var(--jp-content-link-color,#507d00)}';
  host.append(style);
  const label = document.createElement('label'); const caption=document.createElement('span'); caption.textContent='Language: '; label.append(caption);
  const select = document.createElement('select'); select.dataset.courseControl = 'language'; select.setAttribute('aria-label', 'Course language'); label.append(select); host.append(label);
  const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = 'Practice options'; details.append(summary);
  const mode = document.createElement('select'); mode.dataset.courseControl = 'mode'; mode.setAttribute('aria-label', 'Practice mode');
  for (const [value, text] of [['exercises', 'Practice'], ['solutions', 'Prefilled walkthrough']]) mode.add(new Option(text, value));
  details.append(mode); host.append(details);
  const status = document.createElement('p'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); host.append(status);
  const notify = () => {
    if (!host.isConnected) { document.removeEventListener('dli-course-language-state', notify); return; }
    if (shared.value) render(shared.value);
    select.disabled = mode.disabled = shared.busy || shared.unavailable === true;
  };
  document.addEventListener('dli-course-language-state', notify);
  async function request(method, body) {
    const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 65000);
    try {
      const xsrf = document.cookie.split('; ').find(value => value.startsWith('_xsrf='));
      const response = await fetch(endpoint + (method === 'GET' ? '?course=' + encodeURIComponent(course) : ''), {
        method, credentials: 'same-origin', signal: controller.signal,
        headers: {'Content-Type': 'application/json', ...(xsrf ? {'X-XSRFToken': decodeURIComponent(xsrf.slice(6))} : {})},
        ...(body ? {body: JSON.stringify(body)} : {})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || data.reason || 'Language change failed (' + response.status + ')');
      return data;
    } finally { clearTimeout(timer); }
  }
  function render(value) {
    current = value; select.replaceChildren();
    if (!labels[value.language]) throw new Error('Missing widget translation: '+value.language);
    caption.textContent=text('language')+': '; summary.textContent=text('options');
    select.setAttribute('aria-label',text('language')); mode.setAttribute('aria-label',text('options'));
    mode.options[0].text=text('practice');mode.options[1].text=text('solutions');
    for (const language of value.languages) select.add(new Option(language.label, language.value));
    select.value = value.language; mode.value = value.mode; details.hidden = value.modes.length < 2; status.textContent = shared.message;
    select.disabled = mode.disabled = shared.busy || shared.unavailable === true;
    if (shared.backup) {
      const link = document.createElement('a'); link.textContent = text('backup');
      link.href = base + 'files/' + shared.backup.split('/').map(encodeURIComponent).join('/'); link.download = '';
      status.append(' ', link);
    }
  }
  async function change() {
    if (shared.busy || !current) return;
    const selected = {course, language: select.value, mode: mode.value};
    const affected = Array.from(app.shell.widgets('main')).filter(item => item.context && current.notebooks.includes(item.context.path));
    selected.sessions = affected.map(item => item.sessionContext?.session?.id).filter(Boolean);
    if (affected.some(item => shared.pendingRefresh?.includes(item.context.path) && item.context.model.dirty)) {
      status.textContent = text('dirty');
      select.value = current.language; mode.value = current.mode; return;
    }
    if (affected.some(item => item.sessionContext?.session?.kernel?.status === 'busy')) {
      status.textContent = text('busy'); select.value = current.language; mode.value = current.mode; return;
    }
    if (affected.some(item => item.context.model.dirty) && !window.confirm(text('confirm'))) {
      select.value = current.language; mode.value = current.mode; return;
    }
    shared.busy = true; document.dispatchEvent(new Event('dli-course-language-state'));
    const readonly = affected.map(item => [item.context.model, item.context.model.readOnly]);
    try {
      // Resolve the authored widget asset before changing notebook files.
      const sourcePath = [course, 'composer/course_language.js'].filter(Boolean).join('/');
      const asset = await fetch(base + 'files/' + sourcePath.split('/').map(encodeURIComponent).join('/'),
        {credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(15000)});
      if (!asset.ok) throw new Error(text('reopen'));
      const widgetSource = await asset.text();
      status.textContent = text('saving');
      for (const [model] of readonly) model.readOnly = true;
      // Reconcile a previously committed change before saving any stale open model.
      for (const item of affected) {
        if (shared.pendingRefresh?.includes(item.context.path)) {
          await item.context.revert();
          shared.pendingRefresh = shared.pendingRefresh.filter(path => path !== item.context.path);
        }
      }
      for (const item of affected) {
        // RTC autosave can advance the disk revision without updating Context.
        if (item.context.model.collaborative) await item.context.revert();
        await item.context.save();
      }
      const snapshot = item => {
        const notebook = structuredClone(item.context.model.toJSON());
        // Kernel startup can update this field after saving; preserve it below.
        delete notebook.metadata.language_info?.version;
        return JSON.stringify(notebook, (_key, value) => value && typeof value === 'object' && !Array.isArray(value)
          ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value);
      };
      const saved = new Map(affected.map(item => [item, snapshot(item)]));
      shared.pendingRefresh = affected.map(item => item.context.path);
      const result = await request('POST', selected);
      shared.value = result; current = result; shared.unavailable = false;
      shared.message = text('updated'); shared.backup = result.backup;
      for (const item of affected) {
        if (item.context.model.collaborative) {
          // The server refreshes collaborative rooms before replying; wait for delivery.
          const deadline = Date.now() + 10000;
          const expected = result.projected_metadata[item.context.path];
          while (Object.entries(expected).some(([key, value]) => item.context.model.getMetadata(key) !== value)) {
            if (Date.now() >= deadline || item.isDisposed) throw new Error(text('reopen'));
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          // Update Jupyter's contents revision after the collaborative room refresh.
          await item.context.revert();
        } else {
          if (item.context.model.dirty && snapshot(item) !== saved.get(item)) throw new Error(text('newEdits'));
          const version = item.context.model.getMetadata('language_info')?.version;
          await item.context.revert();
          if (version !== undefined) {
            item.context.model.setMetadata('language_info', {...item.context.model.getMetadata('language_info'), version});
          }
        }
        shared.pendingRefresh = shared.pendingRefresh.filter(path => path !== item.context.path);
      }
      // Refresh controls in both open entry notebooks without running learner code.
      for (const item of affected) {
        const selectorCell = () => {
          const cells = item.context.model.cells;
          for (let index = 0; index < cells.length; index++) {
            const candidate = cells.get(index);
            if (candidate.id === 'course-language-switch') return candidate;
          }
        };
        const cell = selectorCell();
        if (!cell) continue;
        if (cell.isDisposed || item.isDisposed) throw new Error(text('reopen'));
        const placeholder = document.createElement('div');
        placeholder.id = 'course-language-' + Array.from(crypto.getRandomValues(new Uint32Array(4))).join('-');
        placeholder.className = 'course-language-control';
        placeholder.setAttribute('role', 'group');
        placeholder.setAttribute('aria-label', text('language'));
        placeholder.textContent = text('loading');
        const nextConfig = {...config, id: placeholder.id, language: result.language};
        const script = widgetSource.replace('__COURSE_' + 'LANGUAGE_CONFIG__', () => JSON.stringify(nextConfig));
        const outputs = [
          {output_type: 'display_data', data: {'text/html': placeholder.outerHTML}, metadata: {}},
          {output_type: 'display_data', data: {'application/javascript': script}, metadata: {}}
        ];
        cell.trusted = true;
        cell.sharedModel.updateOutputs(0, cell.sharedModel.getOutputs().length, outputs);
        if (item.context.model.collaborative) await item.context.revert();
        await item.context.save();
        if (cell.isDisposed || selectorCell() !== cell) throw new Error(text('reopen'));
      }
    } catch (error) {
      shared.message = error.name === 'AbortError' ? text('timeout') : translate(error.message);
      // The server may have committed even if refreshing an open tab failed.
      try { shared.value = await request('GET'); shared.unavailable = false; render(shared.value); }
      catch (_) { shared.unavailable = true; status.textContent = shared.message; select.disabled = mode.disabled = true; }
      if (!host.isConnected) window.alert(shared.message + '\n' + text('reopen'));
    } finally {
      for (const [model, value] of readonly) if (!model.isDisposed) model.readOnly = value;
      await new Promise(resolve => setTimeout(resolve, 1100));
      shared.busy = false; document.dispatchEvent(new Event('dli-course-language-state'));
    }
  }
  select.addEventListener('change', change); mode.addEventListener('change', change);
  try { shared.value = await request('GET'); shared.unavailable = false; render(shared.value); }
  catch (error) { shared.unavailable = true; status.textContent = translate(error.message); select.disabled = mode.disabled = true; }
})();
