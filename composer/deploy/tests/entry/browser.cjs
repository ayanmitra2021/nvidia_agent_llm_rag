const {chromium}=require('playwright');
const fs=require('fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true});let failed=0;
 for(const id of ['course']){
  const base=process.env.COURSE_BASE_URL || 'http://course:8888',context=await browser.newContext(),page=await context.newPage();
  page.setDefaultTimeout(30000);page.on('dialog',d=>d.accept());
  const checks=[];let entrypoint;
  page.on('pageerror',error=>console.error('Browser error:',error.message));
  async function control(){
   const id=await page.evaluate(path=>Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.context?.path===path).id,entrypoint);
   return page.locator('[id="'+id+'"]').locator('.dli-language select[data-course-control="language"]');
  }
  async function openNotebook(target,path){
   await target.waitForFunction(()=>!!window.jupyterapp,null,{timeout:60000});
   const id=await target.evaluate(async path=>{const app=window.jupyterapp;await app.restored;const w=await app.commands.execute('docmanager:open',{path});await w.context.ready;return w.id;},path);
   // Follow Jupyter's kernel picker; context.ready does not mean its socket is connected.
   for(let attempt=0;attempt<3;attempt++){
    await target.waitForFunction(id=>{
     const w=Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.id===id);
     return w.sessionContext.session?.kernel?.connectionStatus==='connected'||Array.from(document.querySelectorAll('[role="dialog"]')).some(d=>d.textContent.includes('Select Kernel'));
    },id,{timeout:60000});
    const dialog=target.getByRole('dialog').filter({hasText:'Select Kernel'});
    if(await dialog.isVisible()){
     await dialog.locator('select').selectOption({label:'Python 3 (ipykernel)'});
     await dialog.locator('button.jp-mod-accept').click();
    }else break;
   }
   await target.waitForFunction(id=>Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.id===id).sessionContext.session?.kernel?.connectionStatus==='connected',id,{timeout:60000});
   await target.waitForFunction(id=>{const kernel=Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.id===id).sessionContext.session?.kernel;return kernel?.connectionStatus==='connected'&&kernel.status==='idle';},id,{timeout:60000});
   return id;
  }
  async function runSelector(){
   const id=await openNotebook(page,entrypoint);
   await page.evaluate(async id=>{
    const app=window.jupyterapp;const widget=Array.from(app.shell.widgets('main')).find(w=>w.id===id);
    widget.content.deselectAll();
    widget.content.activeCellIndex=widget.content.widgets.findIndex(c=>c.model.id==='course-language-switch');
    app.shell.activateById(widget.id);
    await widget.content.scrollToItem(widget.content.activeCellIndex);
    await widget.content.activeCell.ready;
    widget.content.activeCell.editor.focus();
   },id);
   await page.waitForFunction(id=>{const w=window.jupyterapp.shell.currentWidget;return w?.id===id&&w.content.selectedCells.length===1&&w.content.selectedCells[0].model.id==='course-language-switch';},id);
   const ran=await Promise.race([page.evaluate(id=>window.jupyterapp.commands.execute('notebook:run-cell',{widgetId:id}),id),
    new Promise((_,reject)=>{const timer=setTimeout(()=>reject(new Error('Authored selector execution timed out')),60000);timer.unref();})]);
   assert.equal(ran,true);
   await (await control()).waitFor();
   await page.waitForFunction(()=>window.jupyterapp.shell.currentWidget.node.querySelector('.dli-language select')?.options.length>1);
  }
  async function waitIdle(){
   await page.waitForFunction(async()=>{
    const widgets=Array.from(window.jupyterapp.shell.widgets('main')).filter(w=>w.sessionContext);
    const sessions=await(await fetch('/lab/api/sessions')).json();
    return widgets.every(w=>w.sessionContext.session?.kernel?.status==='idle') && sessions.every(s=>s.kernel.execution_state==='idle');
   },null,{timeout:60000});
  }
  async function switchTo(language){
   await waitIdle();
   if (await (await control()).inputValue()===language) return;
   const executions=await page.evaluate(()=>{
    window.fixtureExecutions ||= 0;
    for(const w of window.jupyterapp.shell.widgets('main')){
     const kernel=w.sessionContext?.session?.kernel;
     if(kernel&&!kernel.fixtureObserved){kernel.fixtureObserved=true;kernel.anyMessage.connect((_,{direction,msg})=>{if(direction==='send'&&msg.header.msg_type==='execute_request')window.fixtureExecutions++;});}
    }
    return window.fixtureExecutions;
   });
   await (await control()).click();
   await (await control()).selectOption(language);
   await page.waitForFunction(({path,expected})=>{const w=Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.context?.path===path);const s=w.node.querySelector('.dli-language select');return s?.value===expected&&!s.disabled&&window.__dliCourseLanguageStates?.['']?.value?.language===expected;},{path:entrypoint,expected:language},{timeout:80000});
   assert.equal(await page.evaluate(()=>window.fixtureExecutions),executions,'Language redraw must not execute notebook code');
  }
  try{
   await page.goto(base+'/lab/course-home?token=foyer-browser-fixture');
   await page.locator('#enter:enabled').waitFor();
   const state=await page.evaluate(async()=>await(await fetch('/lab/api/course/language')).json());
   entrypoint=state.entrypoint;
   const initial=state.languages.find(x=>x.value!=='en').value;
   await page.selectOption('#language',initial);await page.click('#enter');
   await runSelector();
   assert.equal(await (await control()).inputValue(),initial);
   await switchTo('en');checks.push('authored selector cell: initial language to English, production startup');
   await page.waitForFunction(async()=>{
    const widget=window.jupyterapp.shell.currentWidget;
    const file=await(await fetch('/lab/api/contents/'+widget.context.path)).json();
    const cell=file.content?.cells.find(c=>c.id==='course-language-switch');
    return cell?.outputs?.some(o=>o.data?.['text/html']) && cell.outputs.some(o=>o.data?.['application/javascript']) && file.content.metadata.course_language==='en';
   },null,{timeout:15000});
   await page.reload();
   await runSelector();
   await switchTo(initial);await switchTo('en');
   assert.equal(await page.evaluate(()=>window.jupyterapp.shell.currentWidget.node.querySelectorAll('.dli-language select[data-course-control="language"]').length),1);
   checks.push('saved HTML and JavaScript outputs; reload and reversal through authored selector cell');
   // Returning in the same browser tab must have a usable recovery path.
   await page.goto(base+'/lab/course-home');await page.locator('#enter:enabled').waitFor();
   await page.selectOption('#language',initial);await page.click('#enter');
   await page.locator('#recovery:not([hidden]) a').waitFor();
   assert.equal((await page.evaluate(async()=>await(await fetch('/lab/api/course/language')).json())).language,'en');
   await page.click('#return-notebook');await runSelector();await switchTo(initial);await switchTo('en');
   for (const language of state.languages) {await switchTo(language.value);await switchTo('en');}
   checks.push('same-tab foyer return, recovery link, all supported languages reverse to English');
   const assessment=await page.evaluate(async()=>{const graph=await(await fetch('/lab/files/composer/translation_web.json')).json();return Object.entries(graph.notebooks).find(([,value])=>value.role==='assessment')?.[0];});
   if(assessment)await openNotebook(page,assessment);
   await page.evaluate(path=>{const app=window.jupyterapp;app.shell.activateById(Array.from(app.shell.widgets('main')).find(w=>w.context?.path===path).id);},entrypoint);
   await switchTo(initial);await switchTo('en');
   checks.push('open assessment does not block language reversal');
   if(!await page.evaluate(()=>window.jupyterapp.shell.currentWidget.context.model.collaborative)) {
    const guard=async route=>{
     if(route.request().method()!=='POST')return route.continue();
     const response=await route.fetch();
     await page.evaluate(()=>{const w=window.jupyterapp.shell.currentWidget;w.context.model.sharedModel.insertCell(w.context.model.cells.length,{cell_type:'markdown',id:'late-learner-note',source:'Keep this late edit',metadata:{}});});
     await route.fulfill({response});
    };
    await page.route('**/api/course/language',guard);
    await(await control()).click();await(await control()).selectOption(initial);
    await page.waitForFunction(expected=>!window.__dliCourseLanguageStates[''].busy&&window.__dliCourseLanguageStates[''].message===expected,state.messages[initial].newEdits);
    assert(await page.evaluate(()=>window.jupyterapp.shell.currentWidget.context.model.toJSON().cells.some(c=>c.id==='late-learner-note')));
    await page.unroute('**/api/course/language',guard);
    // Discard only the injected test edit, then use the normal reload recovery.
    await page.evaluate(()=>window.jupyterapp.shell.currentWidget.context.revert());
    await page.reload();await runSelector();await switchTo('en');
    checks.push('late learner edit remains protected during projection');
   }
   const other=await context.newPage();await other.goto(base+'/lab/lab/workspaces/guard/tree/'+state.entrypoint);
   const otherId=await openNotebook(other,state.entrypoint);
   await other.evaluate(id=>{const widget=Array.from(window.jupyterapp.shell.widgets('main')).find(w=>w.id===id);widget.context.model.sharedModel.insertCell(widget.context.model.cells.length,{cell_type:'markdown',id:'cross-tab-note',source:'Keep my unsaved note',metadata:{}});window.guardWidget=widget;},otherId);
   await waitIdle();
   await (await control()).selectOption(initial);
   await page.waitForFunction(()=>!window.__dliCourseLanguageStates[''].busy&&window.__dliCourseLanguageStates[''].message.includes('other course browser tabs'),null,{timeout:30000});
   assert(await other.evaluate(()=>window.guardWidget.context.model.toJSON().cells.some(c=>c.id==='cross-tab-note')));
   assert.equal((await page.evaluate(async()=>await(await fetch('/lab/api/course/language')).json())).language,'en');
   checks.push('dirty second browser tab remains protected');await other.close();
   await page.evaluate(async()=>{for(const s of window.jupyterapp.serviceManager.sessions.running())await window.jupyterapp.serviceManager.sessions.shutdown(s.id);});
  }catch(error){failed++;checks.push('FAIL: '+error.message);checks.push(await page.evaluate(()=>{const w=window.jupyterapp?.shell.currentWidget;const c=w?.content?.widgets?.find(c=>c.model.id==='course-language-switch');return {dialogs:Array.from(document.querySelectorAll('dialog')).map(d=>d.textContent),kernel:w?.sessionContext?.session?.kernel?.status,sessions:Array.from(window.jupyterapp?.shell.widgets('main')||[]).filter(w=>w.sessionContext).map(w=>({path:w.context.path,status:w.sessionContext.session?.kernel?.status,connection:w.sessionContext.session?.kernel?.connectionStatus})),cell:c&&{id:c.model.id,disposed:c.model.isDisposed,outputs:c.model.outputs?.length,executionCount:c.model.executionCount},text:document.querySelector('.dli-language')?.textContent,shared:window.__dliCourseLanguageStates?.['']&&{busy:window.__dliCourseLanguageStates[''].busy,message:window.__dliCourseLanguageStates[''].message,language:window.__dliCourseLanguageStates[''].value?.language,pendingRefresh:window.__dliCourseLanguageStates[''].pendingRefresh}};}).catch(()=>({})));await page.screenshot({path:'/evidence/course-entry-failure.png',fullPage:true}).catch(()=>{});}
  fs.writeFileSync('/evidence/course-entry.json',JSON.stringify(checks,null,2));console.log('c'+id,checks);await context.close();
 }
 await browser.close();process.exitCode=failed?1:0;
})();
