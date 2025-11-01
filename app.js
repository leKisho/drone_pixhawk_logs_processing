// app.js

// Espera o HTML carregar
document.addEventListener('DOMContentLoaded', () => {
    
    // Pega os elementos da página
    const logSelect = document.getElementById('log-select');
    const tableSelect = document.getElementById('table-select');
    const dataContainer = document.getElementById('data-container');
    const loadingIndicator = document.getElementById('loading');

    const exportButton = document.getElementById('export-button');
    let currentTableName = null; // Guarda o nome da tabela selecionada

    /**
     * Mostra/Esconde o indicador de 'Carregando...'
     */
    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'inline' : 'none';
    }

    /**
     * Busca a lista de Log IDs e preenche o primeiro dropdown
     */
    async function loadLogIds() {
        showLoading(true);
        try {
            const response = await fetch('/api/logs');
            const logIds = await response.json();
            
            logSelect.innerHTML = '<option value="">Selecione um Log</option>'; // Limpa
            logIds.forEach(id => {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = id;
                logSelect.appendChild(option);
            });
        } catch (error) {
            console.error("Erro ao carregar logs:", error);
            logSelect.innerHTML = '<option value="">Erro ao carregar</option>';
        }
        showLoading(false);
    }

    /**
     * Busca a lista de tabelas para um Log ID e preenche o segundo dropdown
     */
    async function loadTables(logId) {
        if (!logId) {
            tableSelect.innerHTML = '<option value="">--</option>';
            tableSelect.disabled = true;
            return;
        }
        
        showLoading(true);
        try {
            const response = await fetch(`/api/tables/${logId}`);
            const tables = await response.json();
            
            tableSelect.innerHTML = '<option value="">Selecione uma Tabela</option>'; // Limpa
            tables.forEach(name => {
                // Mostra o nome "limpo" (ex: "dados_variados")
                const cleanName = name.replace(`_${logId}`, ''); 
                const option = document.createElement('option');
                option.value = name; // Salva o nome real da tabela
                option.textContent = cleanName; // Mostra o nome limpo
                tableSelect.appendChild(option);
            });
            tableSelect.disabled = false;
        } catch (error) {
            console.error("Erro ao carregar tabelas:", error);
            tableSelect.innerHTML = '<option value="">Erro ao carregar</option>';
        }
        showLoading(false);
    }

    /**
     * Busca os dados de uma tabela e renderiza o HTML da tabela
     */
    async function loadTableData(tableName) {
        if (!tableName) {
            dataContainer.innerHTML = '<p style="padding: 20px; text-align: center; color: #888;">Selecione uma tabela.</p>';
            currentTableName = null; // <-- ADICIONE AQUI
            return;
        }

        showLoading(true);
        dataContainer.innerHTML = '';
        
        try {
            const response = await fetch(`/api/data/${tableName}`);
            const data = await response.json(); // Isso é uma lista de objetos

            if (data.length === 0) {
                dataContainer.innerHTML = '<p style="padding: 20px; text-align: center;">Tabela vazia.</p>';
                showLoading(false);
                return;
            }

            // Cria a tabela dinamicamente
            const table = document.createElement('table');
            const thead = document.createElement('thead');
            const tbody = document.createElement('tbody');
            const headerRow = document.createElement('tr');

            // 1. Cria o Cabeçalho (Headers)
            const headers = Object.keys(data[0]);
            headers.forEach(key => {
                const th = document.createElement('th');
                th.textContent = key;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);

            // 2. Cria o Corpo (Linhas de Dados)
            data.forEach(rowObject => {
                const tr = document.createElement('tr');
                headers.forEach(header => {
                    const td = document.createElement('td');
                    let value = rowObject[header];
                    // Arredonda números para 3 casas decimais
                    if (typeof value === 'number') {
                        value = value.toFixed(3);
                    }
                    td.textContent = value;
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });

            table.appendChild(thead);
            table.appendChild(tbody);
            dataContainer.appendChild(table);

            currentTableName = tableName; // Salva o nome da tabela
            exportButton.style.display = 'inline-block'; // Mostra o botão

        } catch (error) {
            console.error("Erro ao carregar dados da tabela:", error);
            dataContainer.innerHTML = '<p style="padding: 20px; color: red;">Erro ao carregar dados.</p>';
        }
        showLoading(false);
    }

    // --- Event Listeners (Gatilhos) ---

    // 1. Quando a página carregar, preencha o primeiro dropdown
    loadLogIds();

    // 2. Quando o usuário mudar o Log, preencha o segundo dropdown
    logSelect.addEventListener('change', (e) => {
        dataContainer.innerHTML = '<p style="padding: 20px; text-align: center; color: #888;">Selecione uma informação.</p>';
        exportButton.style.display = 'none'; // <-- ADICIONE AQUI
        loadTables(e.target.value);
    });

    // 3. Quando o usuário mudar a Tabela, carregue os dados
    tableSelect.addEventListener('change', (e) => {
        exportButton.style.display = 'none'; // <-- ADICIONE AQUI
        loadTableData(e.target.value);
    });

    // 4. Quando o usuário clicar em Exportar
    exportButton.addEventListener('click', () => {
        if (currentTableName) {
        // Chama o novo endpoint do Flask
        window.location.href = `/api/export/${currentTableName}`;
        }
    });


});